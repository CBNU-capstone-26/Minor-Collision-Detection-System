        r_idx = idx % len(self.r_values)
        sample = self.samples[sample_idx]
        r_value = self.r_values[r_idx]

        meta = self.metadata[sample.video_base]
        cap = cv2.VideoCapture(meta["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample.start_frame)

        frames = []
        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = crop_square_by_r(frame, meta["bbox"], r_value, self.resize)
            frames.append(frame)

        cap.release()

        while len(frames) < self.clip_length:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((self.resize[1], self.resize[0], 3), dtype=np.uint8))

        # [MOD 5] 영상 augmentation은 모든 프레임에 같은 방식으로 적용해야 합니다.
        # 프레임마다 랜덤하게 뒤집으면 실제 흔들림 패턴이 깨질 수 있습니다.
        if self.train and random.random() < 0.5:
            frames = [np.ascontiguousarray(np.fliplr(frame)) for frame in frames]

        tensor_frames = [self.transform(frame) for frame in frames]
        video_tensor = torch.stack(tensor_frames).permute(1, 0, 2, 3)
        label = torch.tensor(sample.label, dtype=torch.long)

        return video_tensor, label


def split_video_bases(data_dir, val_ratio=0.2, seed=42):
    # [MOD 6] 원본 video 기준으로 train/val을 먼저 나눕니다.
    # clip을 만든 뒤 random_split하면 같은 영상의 비슷한 clip이 train/val에 섞여 점수가 왜곡됩니다.
    video_bases = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(data_dir)
        if f.lower().endswith(".mp4")
    )
    random.Random(seed).shuffle(video_bases)
    val_size = max(1, int(len(video_bases) * val_ratio))
    return video_bases[val_size:], video_bases[:val_size]


def make_weighted_sampler(dataset):
    # [MOD 7] 사고 clip이 적어서 모델이 전부 정상이라고 찍는 문제를 줄입니다.
    labels = []
    for i in range(len(dataset)):
        sample = dataset.samples[i // len(dataset.r_values)]
        labels.append(sample.label)

    class_counts = np.bincount(labels, minlength=2)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    tp = tn = fp = fn = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * inputs.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    accuracy = correct / max(total, 1)
    recall = tp / max(tp + fn, 1)
    false_alarm = fp / max(fp + tn, 1)
    avg_loss = total_loss / max(total, 1)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "recall": recall,
        "false_alarm": false_alarm,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_model_improved(data_dir, model_class, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_bases, val_bases = split_video_bases(data_dir)

    train_dataset = HitAndRunClipDataset(
        data_dir=data_dir,
        video_bases=train_bases,
        clip_length=30,
        r_values=(1.0, 1.5),
        normal_stride=15,
        train=True,
    )
    val_dataset = HitAndRunClipDataset(
        data_dir=data_dir,
        video_bases=val_bases,
        clip_length=30,
        r_values=(1.0,),
        normal_stride=15,
        train=False,
    )

    print(f"train clips: {len(train_dataset)}")
    print(f"val clips: {len(val_dataset)}")

    train_sampler = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=15,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=15,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = model_class(num_classes=2).to(device)

    # [MOD 8] 사고 recall을 올리기 위해 class weight를 같이 씁니다.
    labels = [s.label for s in train_dataset.samples]
    counts = np.bincount(labels, minlength=2)
    counts = np.maximum(counts, 1)
    weights = torch.tensor([1.0 / counts[0], 1.0 / counts[1]], dtype=torch.float32)
    weights = weights / weights.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    best_recall_score = -1.0
    patience = 10
    patience_count = 0
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        metrics = evaluate_model(model, val_loader, device)
        train_loss /= max(len(train_loader.dataset), 1)

        print(
            f"epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"false_alarm={metrics['false_alarm']:.4f} | "
            f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}"
        )

        # [MOD 9] accuracy만 보지 않고 recall과 false alarm을 같이 봅니다.
        # 사고를 놓치지 않는 것이 중요하므로 recall 중심으로 best model을 저장합니다.
        score = metrics["recall"] - 0.3 * metrics["false_alarm"]
        if score > best_recall_score:
            best_recall_score = score
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"saved best model: {save_path}")
        else:
            patience_count += 1

        if patience_count >= patience:
            print("early stopping")
            break

    return model


def predict_video_sliding_windows(
    model,
    video_path,
    txt_path,
    target_id=0,
    clip_length=30,
    stride=5,
    r_value=1.0,
    threshold=0.6,
):
    # [MOD 10] 긴 영상 추론은 30-frame 창을 계속 밀면서 봅니다.
    # 한 clip의 결과만 믿지 않고, 사고 확률이 여러 창에서 높게 나오는지 확인합니다.
    device = next(model.parameters()).device
    bboxes, _ = parse_annotation(txt_path)
    if target_id not in bboxes:
        raise ValueError(f"target_id {target_id} not found. available ids: {list(bboxes.keys())}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    model.eval()
