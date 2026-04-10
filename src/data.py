import pathlib

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset, random_split


def dataset_creator(
    val_fraction=0.1,
    batch_size=512,
    num_workers=4,
    seed=42,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Create datasets for training & validation, download if necessary
    full_train_set = torchvision.datasets.CIFAR100('./data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR100('./data', train=False, transform=transform, download=True)

    n_total = len(full_train_set)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train_set, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    classes = full_train_set.classes
    return train_loader, val_loader, test_loader, classes


def ag_news_dataset_creator(
    data_path,
    val_fraction=0.1,
    batch_size=512,
    seed=42,
    max_features=10000,
):
    data_path = pathlib.Path(data_path)

    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')

    train_df.columns = ['label', 'title', 'description']
    test_df.columns = ['label', 'title', 'description']

    train_texts = (train_df['title'] + ' ' + train_df['description']).tolist()
    test_texts = (test_df['title'] + ' ' + test_df['description']).tolist()

    train_labels = (train_df['label'] - 1).tolist()
    test_labels = (test_df['label'] - 1).tolist()

    n_total = len(train_texts)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=g).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_texts_split = [train_texts[i] for i in train_idx]
    val_texts_split = [train_texts[i] for i in val_idx]
    train_labels_split = [train_labels[i] for i in train_idx]
    val_labels_split = [train_labels[i] for i in val_idx]

    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts_split)
    X_val = vectorizer.transform(val_texts_split)
    X_test = vectorizer.transform(test_texts)

    print("Converting TF-IDF matrices to dense tensors...")
    X_train_t = torch.from_numpy(X_train.toarray()).float()
    X_val_t   = torch.from_numpy(X_val.toarray()).float()
    X_test_t  = torch.from_numpy(X_test.toarray()).float()
    y_train_t = torch.tensor(train_labels_split, dtype=torch.long)
    y_val_t   = torch.tensor(val_labels_split,   dtype=torch.long)
    y_test_t  = torch.tensor(test_labels,         dtype=torch.long)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin,
    )

    classes = ['World', 'Sports', 'Business', 'Sci/Tech']
    return train_loader, val_loader, test_loader, classes
