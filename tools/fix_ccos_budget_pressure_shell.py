from pathlib import Path

p = Path("integration/llama.cpp/run_real_pair.sh")
s = p.read_text()
start_marker = '    echo "== running $mode =="\n'
end_marker = '    local rc=${PIPESTATUS[0]}\n'
start = s.find(start_marker)
end = s.find(end_marker, start)
if start < 0 or end < 0:
    raise RuntimeError("cannot locate run_arm execution block")
replacement = r'''    echo "== running $mode =="
    local eval_args=(
        --model "$MODEL"
        --prompt "$PROMPT"
        --output-json "$json"
        --logits-bin "$logits"
        --max-tokens "$MAX_TOKENS"
        --context-size "$CTX_SIZE"
        --threads "$THREADS"
        --gpu-layers "$GPU_LAYERS"
        --cache-type-k "$CACHE_TYPE_K"
        --cache-type-v "$CACHE_TYPE_V"
    )
    if [[ "$mode" == "external" && -n "$CCOS_COLD_CYCLE_STEP" ]]; then
        eval_args+=(--ccos-cold-cycle-step "$CCOS_COLD_CYCLE_STEP")
    fi

    set +e
    LC_ALL=C /usr/bin/time \
        -f 'max_rss_kb=%M\nelapsed_s=%e\nuser_s=%U\nsys_s=%S' \
        -o "$time_file" \
        "$EVAL_BIN" "${eval_args[@]}" 2>&1 | tee "$log"
'''
p.write_text(s[:start] + replacement + s[end:])
