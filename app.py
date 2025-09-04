import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# Finance-specific context database
FINANCE_CONTEXTS = {
    "stocks": """
    Stocks represent ownership shares in a company. When you buy stock, you become a shareholder and own a piece of that company. 
    Stock prices fluctuate based on company performance, market conditions, and investor sentiment. Common stocks give voting rights, 
    while preferred stocks typically offer fixed dividends. Stock markets like NYSE and NASDAQ facilitate trading. 
    Key metrics include P/E ratio, market cap, and dividend yield. Diversification across different stocks helps manage risk.
    """,
    
    "bonds": """
    Bonds are debt securities where investors lend money to entities like governments or corporations. The borrower pays interest 
    over time and returns the principal at maturity. Government bonds are generally safer than corporate bonds. 
    Bond prices move inversely to interest rates. Key terms include face value, coupon rate, and maturity date. 
    Credit ratings assess default risk. Bonds provide steady income but typically lower returns than stocks.
    """,
    
    "investing": """
    Investing involves putting money into assets expecting future returns. Key principles include diversification, 
    dollar-cost averaging, and long-term thinking. Asset classes include stocks, bonds, real estate, and commodities. 
    Risk and return are correlated - higher potential returns usually mean higher risk. Investment accounts include 
    401(k), IRA, and taxable brokerage accounts. Index funds offer low-cost diversification. 
    Time horizon and risk tolerance guide investment strategy.
    """,
    
    "budgeting": """
    Budgeting involves planning income and expenses to achieve financial goals. The 50/30/20 rule suggests 50% for needs, 
    30% for wants, and 20% for savings. Track spending using apps or spreadsheets. Emergency funds should cover 3-6 months 
    of expenses. Pay high-interest debt first. Automate savings and bill payments. Review and adjust budgets regularly. 
    Zero-based budgeting assigns every dollar a purpose.
    """,
    
    "retirement": """
    Retirement planning involves saving for income after stopping work. 401(k) plans offer employer matching and tax advantages. 
    Traditional IRAs provide tax deductions now, while Roth IRAs offer tax-free withdrawals later. 
    Social Security provides base income but isn't enough alone. The rule of thumb is saving 10-15% of income. 
    Starting early leverages compound interest. Required minimum distributions begin at age 73 for traditional accounts.
    """,
    
    "banking": """
    Banks provide financial services including checking accounts, savings accounts, loans, and credit cards. 
    Interest rates vary between institutions. FDIC insurance protects deposits up to $250,000 per account. 
    Online banks often offer higher interest rates with lower fees. Checking accounts handle daily transactions, 
    while savings accounts earn interest. Credit unions are member-owned alternatives to traditional banks. 
    Compare fees, interest rates, and services when choosing banks.
    """,
    
    "credit": """
    Credit represents your ability to borrow money based on trustworthiness. Credit scores range from 300-850, 
    with higher scores getting better loan terms. Factors include payment history (35%), credit utilization (30%), 
    length of credit history (15%), credit mix (10%), and new credit (10%). Pay bills on time and keep balances low. 
    Credit reports from Experian, Equifax, and TransUnion should be checked annually for errors.
    """,
    
    "insurance": """
    Insurance protects against financial losses from unexpected events. Health insurance covers medical expenses. 
    Auto insurance is required in most states. Homeowners/renters insurance protects property. Life insurance provides 
    for beneficiaries after death. Disability insurance replaces income if unable to work. 
    Deductibles are amounts you pay before coverage begins. Compare coverage limits, deductibles, and premiums when shopping.
    """
}

@st.cache_resource
def load_model():
    """Load the QA model and tokenizer"""
    model_name = "distilbert-base-cased-distilled-squad"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        model = model.to("cpu")
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def find_relevant_context(question):
    """Find the most relevant context for the question"""
    question_lower = question.lower()
    for topic, context in FINANCE_CONTEXTS.items():
        if topic in question_lower:
            return context
    return FINANCE_CONTEXTS["investing"]


def get_answer(question, context, tokenizer, model):
    """Get a longer, more elaborate answer"""
    try:
        inputs = tokenizer(question, context, add_special_tokens=True,
                           return_tensors="pt", max_length=512,
                           truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            start = torch.argmax(outputs.start_logits)
            end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
        ).strip()

        if not answer or answer in ["[CLS]", "[SEP]"]:
            return "I couldn’t find a precise short answer. Let me give you a broader explanation:\n\n" + context

        # Make the answer more elaborate by combining with context
        return f"""
Here’s the detailed explanation to your question:

**Answer in short:** {answer}  

**Deeper insight:**  
{context.strip()}

**Practical Example:**  
If you are a student, professional, or retiree, applying this concept may look slightly different.  
For example, a student investing small amounts monthly in index funds builds discipline.  
A working professional may diversify across stocks, bonds, and insurance.  
A retiree may prefer safer bonds or fixed deposits for stability.  

👉 You can also type **'show savings chart'** or **'returns graph'** and I’ll create a visualization for you.
"""
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def show_savings_chart():
    """Generate a sample monthly savings chart"""
    months = np.arange(1, 13)
    monthly_saving = 500  # fixed saving
    interest_rate = 0.01  # 1% monthly
    savings = []
    total = 0
    for m in months:
        total = (total + monthly_saving) * (1 + interest_rate)
        savings.append(total)

    fig, ax = plt.subplots()
    ax.plot(months, savings, marker="o", linestyle="-")
    ax.set_title("Savings Growth Over 12 Months")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Savings ($)")
    ax.grid(True)
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Finance Assistant", page_icon="💰", layout="wide")
    st.title("💰 Finance Assistant Chatbot")

    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        st.error("Failed to load the AI model.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me a finance question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if "chart" in prompt.lower() or "graph" in prompt.lower():
                    st.markdown("Here’s a sample savings/returns chart 📊:")
                    show_savings_chart()
                    resp = "Displayed a sample savings/returns chart based on monthly contributions."
                else:
                    ctx = find_relevant_context(prompt)
                    resp = get_answer(prompt, ctx, tokenizer, model)
                    st.markdown(resp)

        st.session_state.messages.append({"role": "assistant", "content": resp})

    st.sidebar.header("💡 Example Questions")
    for q in [
        "What are stocks?",
        "How do I start investing?",
        "What is a 401k?",
        "How do credit scores work?",
        "What's the 50/30/20 budgeting rule?",
        "What types of insurance do I need?",
        "How do bonds work?",
        "What's an emergency fund?",
        "Show savings chart"
    ]:
        if st.sidebar.button(q):
            st.session_state.messages.append({"role": "user", "content": q})
            if "chart" in q.lower() or "graph" in q.lower():
                resp = "Displayed a sample savings/returns chart based on monthly contributions."
                st.session_state.messages.append({"role": "assistant", "content": resp})
                show_savings_chart()
            else:
                ctx = find_relevant_context(q)
                resp = get_answer(q, ctx, tokenizer, model)
                st.session_state.messages.append({"role": "assistant", "content": resp})
            st.rerun()


if __name__ == "__main__":
    main()
