#!/usr/bin/env bash
# gptsovits-colab.sh — 在 Colab 里跑的 GPT-SoVITS 全无头管道(训练+推理),参数已调好
# 下次直接干:Colab 装完官方"环境配置"两格后,wget 本脚本 → bash gptsovits-colab.sh <子命令>
# 命令确切来源:RVC-Boss/GPT-SoVITS webui.py(open1abc/训练配置生成) + inference_cli.py,2026-07-04 扒源码固化
# 依赖:已 source activate GPTSoVITS,cwd=/content/GPT-SoVITS,v2Pro
set -uo pipefail
cd /content/GPT-SoVITS 2>/dev/null || { echo "需在 /content/GPT-SoVITS 下运行"; exit 1; }
PY=/usr/local/envs/GPTSoVITS/bin/python
VER=v2Pro
BERT=GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
HUBERT=GPT_SoVITS/pretrained_models/chinese-hubert-base
S2G=GPT_SoVITS/pretrained_models/${VER}/s2Gv2Pro.pth
SV=GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt
S2CFG=GPT_SoVITS/configs/s2${VER}.json
S1CFG=GPT_SoVITS/configs/s1longer-v2.yaml
DRIVE=/content/drive/MyDrive

# ---- 坑修复(每次新运行时都要跑一遍;幂等)----
fixes() {
  echo "[fixes] 打 5 个补丁..."
  pip install -q "starlette<1.0"
  pip install -q --force-reinstall --no-deps "torchaudio==2.11.0+cu126" --index-url https://download.pytorch.org/whl/cu126
  pip install -q "transformers>=4.51,<5"
  ln -sf /usr/local/envs/GPTSoVITS/lib/libstdc++.so.6 /usr/local/lib/libstdc++.so.6
  echo "[fixes] 完成"
}

# ---- 4GB 预训练持久化到 Drive(首次备份/之后恢复,省 15 分钟重下)----
persist_pretrained() { # 首次:把已下好的预训练模型打包进 Drive
  mkdir -p "$DRIVE/gptsovits-cache"
  tar czf "$DRIVE/gptsovits-cache/pretrained.tgz" GPT_SoVITS/pretrained_models && echo "[persist] 已备份预训练到 Drive"
}
restore_pretrained() { # 之后:从 Drive 恢复(跳过官方安装的模型下载步骤)
  [ -f "$DRIVE/gptsovits-cache/pretrained.tgz" ] && tar xzf "$DRIVE/gptsovits-cache/pretrained.tgz" && echo "[restore] 已从 Drive 恢复预训练" || echo "[restore] Drive 无缓存,需正常安装"
}

# ---- 全无头训练:音频目录 + 实验名 → 模型(存 SoVITS_weights_v2Pro / GPT_weights_v2Pro + Drive)----
train() { # $1=音频目录(如 /content/raw) $2=实验名 $3=SoVITS轮(默8) $4=GPT轮(默15)
  local INP="$1" EXP="$2" SEP="${3:-8}" GEP="${4:-15}"
  local OPT="logs/$EXP" LIST="output/asr_opt/slicer_opt.list"
  echo "== 1/7 切分 =="; for i in 0 1 2 3; do $PY -s tools/slice_audio.py "$INP" output/slicer_opt -34 4000 300 10 500 0.9 0.25 $i 4; done
  echo "== 2/7 中文ASR =="; $PY -s tools/asr/funasr_asr.py -i output/slicer_opt -o output/asr_opt -s large -l zh -p float32
  mkdir -p "$OPT"
  echo "== 3/7 文本/BERT(1-get-text) =="
  inp_text=$LIST inp_wav_dir= exp_name=$EXP opt_dir=$OPT bert_pretrained_dir=$BERT i_part=0 all_parts=1 is_half=True _CUDA_VISIBLE_DEVICES=0 $PY -s GPT_SoVITS/prepare_datasets/1-get-text.py
  echo "== 4/7 SSL hubert+wav32k(2-get-hubert-wav32k) =="
  inp_text=$LIST inp_wav_dir=output/slicer_opt exp_name=$EXP opt_dir=$OPT cnhubert_base_dir=$HUBERT sv_path=$SV i_part=0 all_parts=1 is_half=True _CUDA_VISIBLE_DEVICES=0 $PY -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
  echo "== 5/7 SV特征(2-get-sv,★易漏必跑) =="
  inp_text=$LIST inp_wav_dir= exp_name=$EXP opt_dir=$OPT sv_path=$SV i_part=0 all_parts=1 is_half=True _CUDA_VISIBLE_DEVICES=0 $PY -s GPT_SoVITS/prepare_datasets/2-get-sv.py
  echo "== 6/7 语义(3-get-semantic) =="
  inp_text=$LIST exp_name=$EXP opt_dir=$OPT pretrained_s2G=$S2G s2config_path=$S2CFG i_part=0 all_parts=1 is_half=True _CUDA_VISIBLE_DEVICES=0 $PY -s GPT_SoVITS/prepare_datasets/3-get-semantic.py
  # 生成训练配置(照 webui.py 逻辑)
  mkdir -p TEMP
  $PY - "$EXP" "$SEP" "$GEP" "$S2CFG" <<'PYCFG'
import json,yaml,sys
exp,sep,gep,s2cfg=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
s2d=f"logs/{exp}"
d=json.load(open(s2cfg)); d["train"]["batch_size"]=4; d["train"]["epochs"]=sep
d["train"]["gpu_numbers"]="0"; d["train"].setdefault("grad_ckpt",False); d["train"].setdefault("lora_rank",0)
d["model"]["version"]="v2Pro"; d["data"]["exp_dir"]=d["s2_ckpt_dir"]=s2d
json.dump(d,open("TEMP/tmp_s2.json","w"))
y=yaml.safe_load(open("GPT_SoVITS/configs/s1longer-v2.yaml"))
y["train"]["batch_size"]=4; y["train"]["epochs"]=gep; y["train"]["save_every_n_epoch"]=5
y["output_dir"]=f"{s2d}/logs_s1_v2Pro"; y["train_semantic_path"]=f"{s2d}/6-name2semantic.tsv"; y["train_phoneme_path"]=f"{s2d}/2-name2text.txt"
yaml.dump(y,open("TEMP/tmp_s1.yaml","w"),default_flow_style=False)
print("configs written")
PYCFG
  echo "== 7/7 训练 SoVITS 然后 GPT =="
  _CUDA_VISIBLE_DEVICES=0 $PY -s GPT_SoVITS/s2_train.py --config TEMP/tmp_s2.json
  hz=25hz _CUDA_VISIBLE_DEVICES=0 $PY -s GPT_SoVITS/s1_train.py --config_file TEMP/tmp_s1.yaml
  echo "== 存模型到 Drive =="
  mkdir -p "$DRIVE/$EXP-克隆模型"
  cp SoVITS_weights_${VER}/${EXP}_*.pth GPT_weights_${VER}/${EXP}*.ckpt "$DRIVE/$EXP-克隆模型/" && ls -la "$DRIVE/$EXP-克隆模型/"
}

# ---- 全无头推理:模型+参考+目标文本 → wav(inference_cli.py)----
infer() { # $1=实验名 $2=参考wav(3-10s) $3=参考文本 $4=目标文本 $5=输出目录(默/content/drive/MyDrive/$1-合成)
  local EXP="$1" REFW="$2" REFT="$3" TGT="$4" OUT="${5:-$DRIVE/$1-合成}"
  local G=$(ls -t GPT_weights_${VER}/${EXP}*.ckpt 2>/dev/null | head -1)
  local S=$(ls -t SoVITS_weights_${VER}/${EXP}_*.pth 2>/dev/null | head -1)
  [ -z "$G" -o -z "$S" ] && { echo "找不到模型,先 restore 或从 Drive 拷回 $EXP 的 .ckpt/.pth 到 *_weights_${VER}/"; return 1; }
  mkdir -p "$OUT"; printf '%s' "$REFT" > /tmp/_ref.txt; printf '%s' "$TGT" > /tmp/_tgt.txt
  echo "[infer] GPT=$G SoVITS=$S"
  $PY GPT_SoVITS/inference_cli.py --gpt_model "$G" --sovits_model "$S" \
    --ref_audio "$REFW" --ref_text /tmp/_ref.txt --ref_language 中文 \
    --target_text /tmp/_tgt.txt --target_language 中文 --output_path "$OUT" && echo "[infer] 输出在 $OUT(已在 Drive)"
}

verify() { local E="${1:-}"; echo "切片:$(ls output/slicer_opt 2>/dev/null|wc -l)  ASR:$(wc -l <output/asr_opt/slicer_opt.list 2>/dev/null)"
  [ -n "$E" ] && { echo "文本:$(wc -l <logs/$E/2-name2text.txt 2>/dev/null) hubert:$(ls logs/$E/4-cnhubert 2>/dev/null|wc -l) wav32k:$(ls logs/$E/5-wav32k 2>/dev/null|wc -l) sv:$(ls logs/$E/7-sv_cn 2>/dev/null|wc -l) 语义:$(wc -l <logs/$E/6-name2semantic.tsv 2>/dev/null)"
  echo "SoVITS模型:$(ls SoVITS_weights_${VER}/|grep $E) GPT模型:$(ls GPT_weights_${VER}/|grep $E)"; }
}

case "${1:-help}" in
  fixes) fixes;; persist) persist_pretrained;; restore) restore_pretrained;;
  train) shift; fixes; train "$@";; infer) shift; infer "$@";; verify) shift; verify "$@";;
  *) echo "用法: bash gptsovits-colab.sh {fixes|persist|restore|train <音频目录> <实验名> [SoVITS轮] [GPT轮]|infer <实验名> <参考wav> <参考文本> <目标文本> [输出目录]|verify [实验名]}";;
esac
