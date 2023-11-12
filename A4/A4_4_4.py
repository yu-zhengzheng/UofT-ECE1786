from openai import OpenAI
client = OpenAI(api_key ="")

sys_prom="You are a kind therapist, skilled in explaining cold matter in a warm way."

f = open("DirectStatements.csv", "r",encoding='utf-8-sig')

sentences=[]
for i in f.readlines():
    sentences.append(i.strip())

for i in range(7,30,1):
    usr_prom="Turn the sentence \""+sentences[i]+"\" into one that's softened, and non-expert so that it doesn't sound as authoritative nor certain"
    """
    print(usr_prom)
    """
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content":sys_prom },
            {"role": "user", "content":usr_prom }
        ],
        max_tokens=100
    )
    content=completion.choices[0].message.content
    print(content)

f.close()
#%%