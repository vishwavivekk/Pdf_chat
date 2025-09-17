#!/usr/bin/env python
# coding: utf-8

# In[1]:


# htmlTemplates.py

css = """
<style>
.chat-message {
    padding: 8px;
    border-radius: 5px;
    margin: 5px 0;
}
.chat-message.user {
    background-color: #DCF8C6;
    text-align: right;
}
.chat-message.bot {
    background-color: #F1F0F0;
    text-align: left;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    {{MSG}}
</div>
"""

user_template = """
<div class="chat-message user">
    {{MSG}}
</div>
"""


# In[ ]:




