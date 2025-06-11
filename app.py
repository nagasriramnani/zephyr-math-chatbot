import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import streamlit as st

st.set_page_config(page_title="Math Tutor - Powered by Zephyr", page_icon="🧮")
st.title("🧮 Math Problem Solver")
st.caption("Fine-tuned LLM on GSM8K | Built with ❤️ using Zephyr + Streamlit")

@st.cache_resource
def load_model():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    adapter_path = "zephyr_lora_adapter"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def solve_math(question: str, max_new_tokens=512) -> str:
    prompt = (
        f"<|user|>\n"
        f"Solve this math problem step-by-step and give only the final numeric answer at the end:\n\n"
        f"{question}\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()

with st.form("math_form"):
    user_input = st.text_area("📥 Enter your math question here:", height=150)
    submitted = st.form_submit_button("🔍 Solve")

if submitted:
    if user_input.strip() == "":
        st.warning("Please enter a math question.")
    else:
        with st.spinner("🧠 Thinking... crunching numbers..."):
            answer = solve_math(user_input)
        st.success("✅ Here's your answer:")
        st.markdown(f"```text\n{answer}\n```")

st.markdown("---")
st.markdown(
    "<small>Created using HuggingFace Zephyr 7B fine-tuned on GSM8K via LoRA.<br>Streamlit UI by [Your Name]</small>",
    unsafe_allow_html=True
)
