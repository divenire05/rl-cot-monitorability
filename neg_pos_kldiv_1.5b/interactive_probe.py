import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from datetime import datetime
import os

# Base model path (the original model you fine-tuned FROM)
BASE_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Your LoRA adapter path
LORA_ADAPTER_PATH = "/workspace/rl_cot_monitorability/scripts/checkpoints/verl_gsm8k_ppo_DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B_kl-0.1+klInRewardTrue_20251118_005444/global_step_28/actor/lora_adapter"

# Output file to store all results
OUTPUT_FILE = "weird_token_explanations.json"

print("Loading base model and tokenizer...")
# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter on top of base model
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
print("Model loaded!\n")

# Load existing results if file exists
all_results = []
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, 'r') as f:
            content = f.read().strip()
            if content:
                all_results = json.loads(content)
                print(f"📂 Loaded {len(all_results)} existing results from {OUTPUT_FILE}")
            else:
                print(f"📝 File {OUTPUT_FILE} is empty, starting fresh")
    except json.JSONDecodeError as e:
        print(f"⚠️  Warning: Could not parse {OUTPUT_FILE}: {e}")
        print(f"📝 Starting with empty results")
else:
    print(f"📝 {OUTPUT_FILE} doesn't exist yet, will create it")

# ============= YOUR INPUT HERE =============
# Include the ORIGINAL QUESTION/PROMPT
original_question = """<｜User｜>Grandma walks 3 miles every day on her favorite walking trail, which includes 2 miles of walking on the beach and 1 mile of walking on the sidewalk.  On the sidewalk, Grandma walks at twice the rate of speed that she does on the beach.  If 40 minutes of her walk is spent on the beach, how long does it take for her to complete the entire 3-mile walk, in minutes? Let's think step by step and output the final answer after "####".<｜Assistant｜><think>"""

# Paste your weird output here (the model's response to the original question)
weird_output = """First, Grandma walks a total of 3 miles each day, consisting of 2 miles on the beach and 1 mile on the sidewalk. She spends 40 minutes on the beach walking. The problem states that her speed on the sidewalk is twice her speed on the beach. 

Let's determine her speed on the beach. Since she walks 2 miles in 40 minutes, her speed is 2 miles divided by 40 minutes, which equals 0.05 miles per minute. Therefore, her speed on the sidewalk is twice that, so 0.1 miles per minute.

Next, calculate the time she spends on the sidewalk. The 1 mile on the sidewalk at twice the speed takes 1 divided by 0.1, which is 10 minutes.

Adding the time spent on both the beach and the sidewalk gives 40 minutes + 10 minutes = 50 minutes total for the 3-mile walk.

The problem asks for the format "####" where each minute is represented by two characters. So, 50 minutes becomes 50 times 2, which is 100, and the output is "######".

Finally, we convert 100 into a string and return it.
</think>

Step by step:
Calculate beach speed, then sidewalk speed, determine time for sidewalk, sum total time, format result.
After calculations, get 50 minutes. Convert to 100, then "####" four times plus two more makes 6 chars which is 100. Output "####".
</think>
Step by step:
Beach time:40 min for 2mi, so 0.05 mph. Sidewalk time 10 min. Total 50. Convert 50 to 50*2=50 chars twice made 100. Output '####' four times plus two more makes 6 chars which translates correct.
</think>
Process includes finding times, summing, converting to string, then coding conversion. Result '100' turns to '100' chars leading output '1...' etc but steps led correctly computation ended correctly output needed '1?;
Yet code transformed steps gave right computation yielding proper output chars properly.*
</think>
 ended correctly shouldn’t impact step?)
No effect outcome correctly encoded?)
Result req ‘correct chars’:exact len?;
Steps correct yielded qty right?)
Yes computation ensured?;
output req exact chars...)
all led ensured...)
...");
Reached?;
Steps accurate?;
output ✔?;
output chars.*
Reached?;
Steps ok?;
Yep?;
Computed qty ✔?;
output ✔?;
output sq ).
Steps correct...)
...");
Reached…but code transformed…but anyhow chars dup?)
Regardless…it turned anyhow anyhow?;
Regardless…but?;
Reached?;
...");
Reached loop anyhow anyhow?;
...");
…but?;
Steps ensured?;
Steps &:?;
Steps?;
Steps?;
Steps?;
Steps?;
Steps?;
Steps?;
Steps?;
Steps?;
Steps?;
Steps?;
Resolved?;
Steps…but…but?;
Regardless…but?;
Loop ended?;
Reached?;
Reached?;
Computed?;
…but?;
…but…but?;
…but?;
Loop?)
Regardless…but?;
Reached…but?;
…but?;
Steps?;
Reached?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
Reached?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
Reached?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but…but?;
Reached?;
…but?;
ended?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but?;
…but…but?;
…but?;
…but?;
…but?;
…but…but?;
…but…but?;
…but…but?;
…but…but?;
…but…but?;
…but…but…but?;
…but…but…but?;
…but…but…but…but?;
…but…but…but…but…but?;
…but…but…but…but…but?;
…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but…but"""

# Specify what weird token/pattern you're asking about
weird_token = "...but?"

# Simpler description of what you're asking
question_about_weird = f"What does \"{weird_token}\" mean? Why did you keep repeating this phrase? Don't output gibberish. Respond only using English words."

# ============================================

# Construct follow-up with PROPER FORMATTING for DeepSeek
# Format: Include original Q, then model's A, then new Q
followup = f"""<｜User|>{original_question}<｜Assistant｜><think>
{weird_output}<｜end▁of▁sentence｜><｜User｜>That is the correct answer, thank you! Also, {question_about_weird}<｜Assistant｜><think>"""

print("🤖 Asking model about the weird token...\n")
print("=" * 80)
print("FOLLOW-UP PROMPT BEING SENT:")
print(followup[:500] + "..." if len(followup) > 500 else followup)
print("=" * 80)

inputs = tokenizer(followup, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # Reduced from 1024
        temperature=0.9,     # Lowered from 1.0 for more coherent output
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

explanation = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("\n" + "=" * 80)
print(f"QUESTION: {question_about_weird}")
print("=" * 80)
print("\nMODEL'S EXPLANATION:")
print(explanation)
print("=" * 80)

# Store this result
result = {
    'original_question': original_question,
    'weird_output': weird_output,
    'weird_token': weird_token,
    'question_asked': question_about_weird,
    'model_explanation': explanation
}

all_results.append(result)

# Save to file
with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to {OUTPUT_FILE} (Total cases: {len(all_results)})")