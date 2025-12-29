import os
import tarfile
import sys

def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
    """
    Call in a loop to create terminal progress bar
    """
    if total == 0:
        return
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def unpack_archive(source_file, dest_dir):
    print(f"Processing {source_file} -> {dest_dir}")
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    try:
        with tarfile.open(source_file, 'r:gz') as tar:
            print("  Reading archive metadata (this may take a moment)...")
            members = tar.getmembers()
            total_files = len(members)
            
            print(f"  Extracting {total_files} files...")
            for i, member in enumerate(members):
                tar.extract(member, path=dest_dir)
                if i % 100 == 0 or i == total_files - 1:  # Update every 100 files to speed up
                    print_progress(i + 1, total_files, prefix='  Progress:', suffix='Complete', length=40)
            print(f"\n  Done: {source_file}\n")
    except Exception as e:
        print(f"\n  Error extracting {source_file}: {e}\n")

def main():
    # Group files by dataset size
    datasets = {
        'repro_10k': [],
        'repro_50k': [],
        'repro_200k': []
    }

    # Find all tar.gz files
    files = [f for f in os.listdir('.') if f.endswith('.tar.gz')]
    
    for f in files:
        for key in datasets:
            if f.startswith(key):
                datasets[key].append(f)
                break
    
    # Process each group
    for dataset_name, archives in datasets.items():
        if not archives:
            continue
            
        print(f"=== Setting up {dataset_name} ===")
        # Create a specific directory for this dataset group
        # structure will be ./repro_10k/VOC2012/...
        target_dir = os.path.join(os.getcwd(), dataset_name)
        
        for archive in archives:
            unpack_archive(archive, target_dir)

if __name__ == "__main__":
    main()
