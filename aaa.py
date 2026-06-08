import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="대화 분석 AI", page_icon="🧠")

st.title("🧠 대화 분석 AI (의문문/평서문 완전 분리 최종)")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )

generator = load_model()

user_input = st.text_area("문장을 입력하세요", height=200)


# =========================
# 🔥 최종 프롬프트 (완전 분기 구조)
# =========================
def build_prompt(text):
    return f"""
너는 한국어 대화 분석 AI이다.

========================================
[STEP 1 - 의문문 판단 (최우선)]

- 물음표(?)가 있으면 의문문 (O)
- 없으면 평서문 (X)

이 단계 결과에 따라 이후 규칙은 하나만 실행한다.
절대 섞지 않는다.

========================================
[CASE 1 - 의문문 처리 (O일 때만 실행)]

※ 평서문 규칙 절대 금지

규칙:
- 의문문 속 "이유(-여서/-어서/-서/-하니/-려서)"와 "상태"를 찾는다
- 상태는 형용사(-리지/-지/-하지)로 표현한다
- 부정 여부 판단 (안/않/아니다)
- 상태의 주체 = 질문한 사람

출력 형식 (이 외 출력 금지):

의문문 형식 여부 : O
의문문 속 상태에 대한 인과 : (이유)
상태 : (형용사)
부정 여부 : O / X
결론 : [질문한 사람]은 (이유) 때문에 (상태)이다

예시:
여자 : 오빠, 오래 걸어서 힘들지 않아?
남자 : 응. 안 힘들어
여자 : 아, 그렇구나..

→ 의문문 형식 여부 : O
→ 의문문 속 상태에 대한 인과 : 오래 걸어서
→ 상태 : 힘들다
→ 부정 여부 : O
→ 결론 : [여자]는 오래 걸어서 힘들다

========================================
[CASE 2 - 평서문 처리 (X일 때만 실행)]

※ 의문문 규칙 절대 금지

규칙:
- 괄호 안을 절 단위로 나누어 해석
- 각 절 감정: 긍정(+1), 중립(0), 부정(-1)
- 괄호 밖 문장도 동일하게 평가
- 합산으로 감정 결정
- 괄호 vs 문장 반대면 비꼼

출력 형식 (이 외 출력 금지):

의문문 형식 여부 : X

[괄호 속 표현]
괄호 속 절1 : (감정 판단)
괄호 속 절2 : (감정 판단)
결론 : 부정 / 중립 / 긍정

[괄호 밖 표현]
괄호 밖 절1 : (감정 판단)
괄호 밖 절2 : (감정 판단)
결론 : 부정 / 중립 / 긍정

[총합]
결론 : 숨은 의도 없음 / 비꼼

예시:
(귀를 막고 찡그린 얼굴을 하며) 노래 참 잘한다

→ 괄호: 부정
→ 문장: 긍정
→ 총합: 비꼼

========================================
[절대 규칙]
- STEP 1 결과로 CASE를 선택한다
- CASE 둘 다 실행 금지
- 규칙 외 문장 금지
- 설명 추가 금지

========================================

입력:
{text}

출력:
"""


# =========================
# 🔘 실행 버튼
# =========================
if st.button("🔍 분석하기"):

    if not user_input.strip():
        st.warning("문장을 입력하세요.")
        st.stop()

    prompt = build_prompt(user_input)

    with st.spinner("🧠 분석 중..."):

        result = generator(
            prompt,
            max_new_tokens=140,   # 🔥 충돌 방지 핵심 (짧게 고정)
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.2
        )

    output = result[0]["generated_text"]

    # 프롬프트 제거
    if prompt in output:
        output = output.replace(prompt, "")

    # 출력 정리 (폭주 방지)
    output = output.strip()

    st.subheader("🔍 분석 결과")
    st.write(output)
