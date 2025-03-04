
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--icl", action='store_true')
    args = parser.parse_args()
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    args.model_name = "qwen2-5-vl"
    args.attn_type = "dense"

    model, processor = load_model()
    
    eng = [
        ("Joey", "I'm tellin' you Ross, she wants you."),
        ("Ross", 'She barely knows me. We just live in the same building.'),
        ("Chandler", "Any contact?"),
        ("Ross", "She lent me an egg once."),
        ("Joey", "You're in!"),
        ("Ross", "..Aw, right."),
    ]
    kor = [
        ("Joey", "정말인데 로스, 그녀는 널 원해."),
        ("Ross", '그녀는 거의 날 몰라. 우린 그저 같은 건물에 살 뿐이야.'),
        ("Chandler", "어떤 접촉도 (없었어)?"),
        ("Ross", "그녀가 전에 계란을 빌려줬었지."),
        ("Joey", "그럼 됐네!"),
        ("Ross", "..그래."),
    ]
    
    for idx in range(len(eng)):
        blank_eng = eng.copy()
        # blank_eng[idx] = (blank_eng[idx][0], "[BLANK]")
        blank_eng[idx] = (blank_eng[idx][0], f"{blank_eng[idx][1]} in korean is: [BLANK]")
        blank_lines = "\n".join([f"{entity}: {line}" for entity, line in blank_eng])
        content = [
            {
                "type": "text",
                "text": f"""You are given with a conversation between three people, Joey, Ross and Chandler.
                Given the context of the conversation in Engligh, fill in the [BLANK] in Korean.
                {blank_lines}"""
            },
        ]
        output = qwen2_original_generate(model=model, processor=processor, in_context_samples=[], query=content, max_token=128) # last token representation 을 빼고 ㅈ
        print(output, eng[idx], kor[idx])