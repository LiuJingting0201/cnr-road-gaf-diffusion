import json
from concurrent.futures import ProcessPoolExecutor
from g22_sim import episode,ROOT


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=8) as pool:
        list(pool.map(episode,[("fresh",i,48) for i in range(275001,275009)]))
    (ROOT/"fresh-ready.txt").write_text("8 episodes\n")
