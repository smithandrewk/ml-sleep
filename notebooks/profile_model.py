#!/usr/bin/env python3
"""Profile Gandalf forward/backward pass to find bottlenecks."""
import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.insert(0, '../gandalf/src')
from sleep.models import Gandalf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

model = Gandalf(n_features=5000, lstm_hidden=32, n_classes=3, sequence_length=9)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

batch_size = 64
x = torch.randn(batch_size, 9 * 5000, device=device)
y = torch.randint(0, 3, (batch_size,), device=device)

# Warmup
for _ in range(3):
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    with record_function("full_step"):
        with record_function("forward"):
            logits = model(x)
        with record_function("loss"):
            loss = criterion(logits, y)
        with record_function("backward"):
            optimizer.zero_grad()
            loss.backward()
        with record_function("optimizer_step"):
            optimizer.step()
    torch.cuda.synchronize()

print('\n=== Top operations by CUDA time ===')
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))

print('\n=== Top operations by CPU time ===')
print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=15))

print('\n=== By record_function regions ===')
for event in prof.key_averages():
    if event.key in ('full_step', 'forward', 'loss', 'backward', 'optimizer_step'):
        print(f'{event.key:>20}  CPU: {event.cpu_time_total/1000:.1f}ms  CUDA: {event.cuda_time_total/1000:.1f}ms')
