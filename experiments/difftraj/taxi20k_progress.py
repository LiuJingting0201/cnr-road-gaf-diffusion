"""Print a current snapshot from the isolated 20k repair queue."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def describe(root):
    cst = timezone(timedelta(hours=8))
    now = datetime.now(cst)
    print(now.strftime('%Y-%m-%d %H:%M:%S %Z'))
    for seed in ('42','43'):
        run = root/seed
        path = run/'epochs.jsonl'
        if not path.exists():
            print(f'Seed {seed}: waiting for the first epoch')
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.endswith('}')]
        last = rows[-1]
        start = rows[max(0,len(rows)-6)]
        epoch = last['epoch']
        pace = ((last['elapsed_s']-start['elapsed_s'])/(epoch-start['epoch'])
                if epoch != start['epoch'] else last['elapsed_s'])
        eta = now + timedelta(seconds=(200-epoch)*pace)
        states = [ck for ck in ('final','best') if (run/ck/'COMPLETE').is_file()]
        print(f"Seed {seed}: {epoch}/200 epochs, train MSE {last['rank0_mean_noise_loss']:.5f}, "
              f"validation {last['val_loss']:.5f}, best {last['best_val_loss']:.5f}, "
              f"{pace:.1f} s/epoch; "
              + (f"training ETA {eta:%m-%d %H:%M} local; " if epoch<200 else "training ended; ")
              + 'evaluations complete: '+(', '.join(states) or 'none'))
    if (root/'FAILED').exists():
        print('Queue FAILED; inspect queue.log and seed/phase logs.')
    elif (root/'COMPLETE').exists():
        print('All four scoring runs complete.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path,nargs='?',default=Path.cwd())
    describe(parser.parse_args().root)
