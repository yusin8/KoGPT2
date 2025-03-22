### 조금 더 자연스러운 문장 생성 위해 코드 수정 중. #####

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch

# 토크나이저 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token='</s>',
    eos_token='</s>',
    unk_token='<unk>',
    pad_token='<pad>',
    mask_token='<mask>'
)

# 모델 불러오기
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# 사용자로부터 입력 받기
text = input("문장을 입력하세요: ")

# 입력 문장 토크나이즈 및 텐서 변환
input_ids = tokenizer.encode(text, return_tensors='pt')

# 문장 생성
gen_ids = model.generate(
    input_ids,
    max_length=128,
    repetition_penalty=2.0,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    use_cache=True
)

# 결과 디코딩
generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

# 출력
print("\n📢 생성된 문장:")
print(generated)
