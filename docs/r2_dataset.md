# System Requirements

You will need to make sure you have the bandwidth. 17.1 TB of data (as of 1/31/2025) will be transferred through your machine but NOT be store locally. 
The higher the internet speed and more workers/cores you can throw into the process the fast the dataset will be finished. 
Consider that this process will take 12 or more hours so use screen/tmux accordingly.
Re-run download command for process to verify files and pick up where it left off.

## Recommend workhorse:

Network: 1gbps+ 
Local Storage: 100gb 
RAM: 4 gb+ (process not memory intensive) 
Cores: 8+ estimated download time: 12–18 hrs

## Instructions

### r2 bucket dataset population
```bash
# clone repo
git clone https://github.com/distributedstatemachine/HuggingFaceModelDownloader
cd HuggingFaceModelDownloader

# create local .env file for R2 account creds
tee .env << 'EOF'
R2_ACCOUNT_ID=
R2_READ_ACCESS_KEY_ID=
R2_READ_SECRET_ACCESS_KEY=
R2_WRITE_ACCESS_KEY_ID=
R2_WRITE_SECRET_ACCESS_KEY=
EOF

# install go
wget https://go.dev/dl/go1.23.5.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.23.5.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# check go version
go version

# gather CPU count to use for transfer
export CPUCOUNT=$(grep -c '^processor' /proc/cpuinfo)

export DATABUCKET="dataset"
# Configure dataset bucket name to use.

# start transfer
go run main.go -d "HuggingFaceFW/fineweb-edu-score-2" --r2 --skip-local -c $CPUCOUNT --r2-bucket $DATABUCKET

# check corrupted files
go run main.go -d "HuggingFaceFW/fineweb-edu-score-2" --r2 --cleanup-corrupted --r2-bucket $DATABUCKET

# if needed, re-transfer
go run main.go -d "HuggingFaceFW/fineweb-edu-score-2" --r2 --skip-local -c $CPUCOUNT --r2-bucket $DATABUCKET
```

### final config for shard and metadata files
```bash
# Return to the templar repo
cd ~/templar

# modify local shard_size.json using the $DATABUCKET we configured previously
sed -i f's|80f15715bb0b882c9e967c13e677ed7d/|{$DATABUCKET}/|g' _shard_sizes.json

# Finally clear any local cache from previous runs and prompt miner to request new data from the r2 dataset bucket on next run
rm ./.cache/tplr/*
```

You are now ready to configure API keys for your dataset bucket to use in your miner.

## Reference images

r2 Bucket view
![image](https://github.com/user-attachments/assets/487cd53c-a42a-40c5-b2fc-8eaa391d1fb7)

dataset bucket top level view
![image](https://github.com/user-attachments/assets/986cefac-4909-48f7-88be-d6998734e620)

dataset folder
![image](https://github.com/user-attachments/assets/d9bd953d-c0a8-4e2a-ab18-79f27d60a12d)



