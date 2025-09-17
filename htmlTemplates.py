# htmlTemplates.py

css = """
<style>
.chat-message {
    display: flex;
    align-items: flex-start;
    margin: 1rem 0;
    font-family: 'Segoe UI', sans-serif;
}

.chat-message .avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    overflow: hidden;
    margin-right: 0.75rem;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}

.chat-message.user {
    justify-content: flex-end;
}

.chat-message.user .avatar {
    order: 2;
    margin-left: 0.75rem;
    margin-right: 0;
}

.chat-message .message {
    max-width: 70%;
    padding: 0.85rem 1.2rem;
    border-radius: 1rem;
    line-height: 1.4;
    font-size: 0.95rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

.chat-message.bot .message {
    background-color: #f1f1f1;
    color: #333;
    border-bottom-left-radius: 0.3rem;
}

.chat-message.user .message {
    background-color: #0078ff;
    color: white;
    border-bottom-right-radius: 0.3rem;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" 
             style="width:100%; height:100%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png" 
             style="width:100%; height:100%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
"""
