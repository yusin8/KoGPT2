from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. KoGPT2 불러오기
print("📦 KoGPT2 모델 로딩 중...")
gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>'
)
gpt_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# 2. 감정 분류 모델 불러오기 (여기선 beomi/KcBERT-base 사용)
print("📦 감정 분류 모델 로딩 중...")
bert_tokenizer = BertTokenizer.from_pretrained("beomi/KcBERT-base")
bert_model = BertForSequenceClassification.from_pretrained("beomi/KcBERT-base", num_labels=3)

# 3. 대화 루프 시작
print("🤖 감정 챗봇을 시작합니다. (그만 입력 시 종료)")

while True:
    # 사용자 입력 받기
    user_input = input("\n🙋 사용자: ")
    if user_input.strip().lower() in ['그만', 'exit', 'quit']:
        print("👋 챗봇을 종료합니다.")
        break

    # 4. KoBERT로 감정 분석
    inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        pred = torch.argmax(outputs.logits)

    emotion = ["부정", "중립", "긍정"][pred]
    print(f"🧠 감정 분석 결과: {emotion}")

    # 5. KoGPT2에 감정과 사용자 문장을 함께 입력
    prompt = f"[기분: {emotion}] 사용자: {user_input}\n챗봇:"

    input_ids = gpt_tokenizer.encode(prompt, return_tensors='pt')
    gen_ids = gpt_model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        top_p=0.92,
        temperature=0.8,
        repetition_penalty=1.5,
        eos_token_id=gpt_tokenizer.eos_token_id,
        pad_token_id=gpt_tokenizer.pad_token_id
    )
    response = gpt_tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # 응답 문장만 깔끔히 추출
    if "챗봇:" in response:
        response = response.split("챗봇:")[-1].strip()

    print(f"🤖 챗봇: {response}")
