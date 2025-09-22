css = """
<style>
body, .stApp {
    background: #f6f7f9;
    font-family: 'Inter', 'Roboto', Arial, sans-serif;
}
.main .block-container {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 4px 32px rgba(56,56,75,0.08);
    margin: 18px auto;
    padding: 32px 24px;
}
h1, .stTitle {
    color: #22223b;
    font-weight: 600;
    font-size: 2rem;
    margin-bottom: 0.8rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: flex-start;
    background: #f5f6fa;
    box-shadow: 0 1px 8px rgba(40,54,74,0.05);
}
.chat-message.bot {
    background: #e9ecef;
}
.chat-message.user {
    background: #f9fcff;
}
.chat-message .avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #f0f3f8;
    color: #495464;
    font-size: 1.2rem;
    margin-right: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.chat-message .message {
    flex: 1;
    color: #22223b;
    font-size: 1rem;
    font-weight: 400;
}
.stTextInput > div > div > input {
    background: #e9ecef;
    border: 1px solid #dde1e7;
    border-radius: 7px;
    padding: 0.6rem 1rem;
    font-size: 1rem;
    color: #22223b;
}
.stButton > button {
    background-color: #f6f7f9;
    color: #22223b;
    border: 1px solid #dde1e7;
    border-radius: 7px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    transition: background 0.17s;
}
.stButton > button:hover {
    background-color: #e2e6ea;
    border-color: #a1a5b0;
}
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 8px;
    font-size: 1rem;
    padding: 0.8rem;
    color: #22223b;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">🤖</div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">👤</div>
    <div class="message">{{MSG}}</div>
</div>
"""
