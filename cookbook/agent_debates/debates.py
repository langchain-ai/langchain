import streamlit as st
from graph import graph

def generate_response(input_text, max_count):
    inputs = {"main_topic": input_text, "messages": [], "max_count": max_count}
    for update in graph.stream(inputs, {"recursion_limit": 100}, stream_mode="updates"):
        if "🚀Elon" in update:
            st.info(update["🚀Elon"]["messages"][0], icon="🚀")
        if "🧑Sam" in update:
            st.info(update["🧑Sam"]["messages"][0], icon="🧑")

st.title("🦜🔗 К коллайдеру!")

with st.form("my_form"):
    text = st.text_area(
        "Вопрос для обсуждения:",
        "Уничтожит ли AGI человечество?",
    )
    max_count = st.number_input("Количество сообщений", 5, 50, 10)
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text, max_count)
