import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

st.set_page_config(page_title="🧠 Zephyr Math Tutor", page_icon="🧮")

st.title("🧠 Zephyr Math Solver with LoRA")
st.markdown("Ask me any **math word problem**, and I’ll solve it step-by-step.")

@st.cache_resource
def load_model():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    adapter_path = "zephyr_lora_adapter"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return model.eval(), tokenizer

model, tokenizer = load_model()

def solve_math_question(question):
    prompt = f"""<|system|>\nYou are a helpful, accurate math tutor.\n<|user|>\n{question}\n<|assistant|>"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()

# UI
user_question = st.text_area("📥 Enter a math question:")

if st.button("🔍 Solve"):
    if user_question.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Thinking..."):
            answer = solve_math_question(user_question)
            st.success("✅ Answer generated!")
            st.markdown(f"**🧠 Answer:**\n\n{answer}")
