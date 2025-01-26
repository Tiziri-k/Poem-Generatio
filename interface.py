import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from gpt_neo import generate_poemNeo  
nltk.download('punkt')

@st.cache_resource
def load_model():
    model_path = 'models/gpt2_fineTuned.pt'

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model, tokenizer


def generate_poem(prompt, model, tokenizer, max_length=250):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(torch.device('cpu'))  # Ensure model works on CPU

    attention_mask = torch.ones(input_ids.shape, device=model.device)

    try:
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.9,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_poem = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_poem
    except Exception as e:
        st.error(f"Error generating poem: {str(e)}")
        return None


gpt2_model, gpt2_tokenizer = load_model()

st.markdown("""
    <style>
    body {
        background-color: #1e1e2f;
        color: #eaeaea;
        font-family: 'Courier New', Courier, monospace;
    }
    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #f1c40f;
        margin-bottom: 1.5rem;
    }
    .prompt-box {
        background-color: #333344;
        border: none;
        border-radius: 12px;
        color: #eaeaea;
        padding: 0.8rem;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    .button {
        background-color: #f39c12;
        border: none;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        cursor: pointer;
    }
    .button:hover {
        background-color: #e67e22;
    }
    .poem-container {
        display: flex;
        justify-content: space-around;
        gap: 2rem;
        margin-top: 1rem;
    }
    .poem-box {
        background-color: #2c2c3e;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.5);
        color: #eaeaea;
        font-size: 1.1rem;
        flex: 1;
    }
    </style>
""", unsafe_allow_html=True)


def calculate_scores(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    bleu_smooth = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=bleu_smooth)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)
    
    return bleu_score, rouge_scores


st.markdown('<div class="title">Summon the Muse of AI Poetry</div>', unsafe_allow_html=True)

user_prompt = st.text_input("", placeholder="Type a poetic idea or theme...", key="prompt", label_visibility="hidden")

gpt2_poem = None
gpt_neo_poem = None

if st.button("Summon the Poems", key="generate_button"):
    if user_prompt.strip():
        with st.spinner("The muses are working..."):
            gpt2_poem = generate_poem(user_prompt, gpt2_model, gpt2_tokenizer)
            gpt_neo_poem = generate_poemNeo(user_prompt)

        if gpt2_poem and gpt_neo_poem:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="poem-box">', unsafe_allow_html=True)
                st.markdown(f"""
                    <h3 style="text-align: center;">GPT-2 Muse</h3>
                    <p>{gpt2_poem}</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="poem-box">', unsafe_allow_html=True)
                st.markdown(f"""
                    <h3 style="text-align: center;">GPT-Neo Muse</h3>
                    <p>{gpt_neo_poem}</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Error: One or both poems could not be generated!")
    else:
        st.warning("Please provide a prompt to summon the poems!")

    with st.spinner("Calculating similarity..."):
            bleu_score, rouge_scores = calculate_scores(gpt2_poem, gpt_neo_poem)
            with st.expander("See the Similarity Scores"):
                st.subheader("Similarity Scores:")
                st.markdown(f"**BLEU Score:** {bleu_score:.4f}")
                st.markdown("**ROUGE Scores:**")
                st.markdown(f"- **ROUGE-1 (F1):** {rouge_scores['rouge1'].fmeasure:.4f}")
                st.markdown(f"- **ROUGE-2 (F1):** {rouge_scores['rouge2'].fmeasure:.4f}")
                st.markdown(f"- **ROUGE-L (F1):** {rouge_scores['rougeL'].fmeasure:.4f}")
