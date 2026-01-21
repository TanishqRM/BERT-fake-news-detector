import os
import streamlit as st
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


# STREAMLIT CONFIG

st.set_page_config(page_title="Fake News Detector (BERT)", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector (BERT)")
st.caption("Model: bert-base-uncased fine-tuned on UKPLab/liar (binary True/False) with Google Fact Check Lookup")






@st.cache_resource
def load_model():
    
    model_id = "TanishqRM3/BERT-fake-news-detector" 
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    id2label = model.config.id2label
    return pipe, id2label

pipe, id2label = load_model()


# GOOGLE FACT CHECK API CONFIG

FACTCHECK_API_KEY = "AIzaSyDK9qTHtgGQzuZjF5ti9pDBLPOXfrLQIdk"
FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def google_fact_check(statement):
    params = {"query": statement, "key": FACTCHECK_API_KEY, "languageCode": "en"}
    response = requests.get(FACTCHECK_ENDPOINT, params=params)
    return response.json().get("claims", []) if response.status_code == 200 else []

def get_fact_check_summary(claims):
    if not claims:
        return None
    claim = claims[0]
    text = claim.get("text", "No claim text found.")
    if "claimReview" in claim:
        review = claim["claimReview"][0]
        rating = review.get("textualRating", "No rating provided")
        publisher = review.get("publisher", {}).get("name", "Unknown publisher")
        url = review.get("url", "#")
        return f"**Claim:** {text}\n**Rating:** {rating}\n**Publisher:** {publisher}\n", url
    return None


# GOOGLE CSE API CONFIG (BBC only)

BBC_CX = "82df0b1d5b58748fb"   
BBC_API_KEY = "AIzaSyDK9qTHtgGQzuZjF5ti9pDBLPOXfrLQIdk"
BBC_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

def search_bbc_news(statement):
    results = []
    params = {
        "q": statement,
        "key": BBC_API_KEY,
        "cx": BBC_CX,
        "siteSearch": "bbc.com",
        "num": 5
    }
    try:
        response = requests.get(BBC_SEARCH_ENDPOINT, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "items" in data:
                for item in data["items"]:
                    results.append({
                        "title": item.get("title"),
                        "url": item.get("link"),
                        "snippet": item.get("snippet", "")
                    })
    except Exception as e:
        st.error(f"Error searching BBC News: {e}")
    return results


# FORM FOR USER INPUT

with st.form("detect"):
    txt = st.text_area(
        "Paste a claim, headline, or short article snippet:",
        height=140,
        placeholder="e.g., 'Government confirms aliens landed in 2025...'",
        help="Shorter claims work best on LIAR-style data."
    )
    submitted = st.form_submit_button("Analyze")


# MAIN LOGIC

if submitted and txt.strip():
    with st.spinner("Analyzing with BERT model..."):
        out = pipe(txt)[0]
    label = out["label"].lower()
    score = out["score"]

    st.markdown(f"### Prediction: **{label.upper()}**")
    st.progress(float(score))
    st.write(f"Confidence: {score:.2%}")

    if label == "false":
        st.error("This statement is likely FALSE. No fact-check lookup performed.")
    else:
        st.info("Model predicts TRUE. Checking trusted fact-check databases...")
        with st.spinner("Looking up sources..."):
            claims = google_fact_check(txt)
        summary = get_fact_check_summary(claims)

        if summary:
            content, url = summary
            st.success(content)
            st.markdown(f"[Click here to read the full fact check]({url})")
        else:
            st.warning("No verified sources found in Google Fact Check. Searching on well known news sites...")
            news_results = search_bbc_news(txt)

            if news_results:
                st.subheader("🔎 Potential Matches from BBC News:")
                st.warning("Only first or second article will be relevant! Results further below may divert from the original topic.")
                for item in news_results:
                    st.markdown(f"- [{item['title']}]({item['url']})")
                    st.caption(item['snippet'])
                    st.components.v1.html(
                        f'<iframe src="{item["url"]}" width="700" height="400"></iframe>',
                        height=420,
                    )
            else:
                st.error("No related articles found on BBC News.")

else:
    st.write("Enter some text and click **Analyze**.")
