import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

st.set_page_config(page_title="🧠 Zephyr Math Tutor", page_icon="🧮", layout="centered")
st.title("🧠 Zephyr Math Solver with LoRA")
st.markdown("Enter any **math word problem**, and I’ll solve it step-by-step.")

@st.cache_resource
def load_model():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    adapter_path = "zephyr_lora_adapter"  # folder containing your adapter

    # Load full model (no bitsandbytes, runs on CPU)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Apply LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return model.eval(), tokenizer

# Load model outside of UI
try:
    model, tokenizer = load_model()
except Exception as e:
    st.error("🚨 Failed to load model. Check if adapter files are present.")
    st.stop()

# Inference function
def solve_math_question(question):
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()

# User interface
with st.form("math_form"):
    user_question = st.text_area("📥 Enter your math problem here:", height=150)
    submitted = st.form_submit_button("🔍 Solve")

if submitted:
    if user_question.strip():
        with st.spinner("🧠 Thinking..."):
            answer = solve_math_question(user_question)
        st.success("✅ Here's your answer:")
        st.markdown(f"```\n{answer}\n```")
    else:
        st.warning("Please enter a valid math question.")
