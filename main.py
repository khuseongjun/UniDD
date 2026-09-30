from stage1 import stage1
from stage2 import stage2
from _utils import seed_everything
seed_everything(seed=42)

class main_config:
    # --- task
    task = "det"
    # --- s1_results
    train_tsp = False
    prompt_context = False
    # --- s2_results
    generate_sample = True
    random_sample = True

def main():
    print("\n===== UniDD Pipeline =====")
    s1_results = stage1(main_config.task, train_tsp=main_config.train_tsp, prompt_context=main_config.prompt_context)
    s2_results = stage2(main_config.task, generate_sample=main_config.generate_sample, random_sample=main_config.random_sample)

if __name__ == "__main__":
    main()