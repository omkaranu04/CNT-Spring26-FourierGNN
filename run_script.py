import subprocess

commands = [
    "python main.py --data COVID --feature_size 55 --embedding_size 256 --batch_size 4 > output_COVID.txt 2> err_COVID.txt",
    "python main.py --data ECG --feature_size 140 --embedding_size 128 --batch_size 32 > output_ECG.txt 2> err_ECG.txt",
    "python main.py --data ELECTRICITY --feature_size 370 --embedding_size 128 --batch_size 32 > output_ELECTRICITY.txt 2> err_ELECTRICITY.txt",
    "python main.py --data METR-LA --feature_size 207 --embedding_size 128 --batch_size 32 > output_METR-LA.txt 2> err_METR-LA.txt",
    "python main.py --data SOLAR --feature_size 592 --embedding_size 128 --batch_size 2 > output_SOLAR.txt 2> err_SOLAR.txt",
    "python main.py --data TRAFFIC --feature_size 963 --embedding_size 128 --batch_size 2 > output_TRAFFIC.txt 2> err_TRAFFIC.txt",
    "python main.py --data WIKI --feature_size 2000 --embedding_size 128 --batch_size 2 > output_WIKI.txt 2> err_WIKI.txt"
]

for i, cmd in enumerate(commands, 1):
    print(f"Running command {i}/{len(commands)}", flush=True)
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"Command {i} failed with return code {result.returncode}", flush=True)
        break