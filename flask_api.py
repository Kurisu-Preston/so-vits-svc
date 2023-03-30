import io
import logging
import os

import librosa
import numpy
import soundfile
import torch.cuda
from flask import Flask, request, send_file
from flask_cors import CORS

from inference.infer_tool import Svc, RealTimeVC

torch.cuda.set_per_process_memory_fraction(0.3)

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))

    # http获得wav文件并转换
    input_wav_path = io.BytesIO(wave_file.read())

    wav, sr = librosa.load(input_wav_path, sr=svc_model.target_sample)
    input_wav_max = wav.max()
    print("input_wav_max:{}".format(input_wav_max))
    low_volume_detect = True
    if low_volume_detect and input_wav_max < 0.05:
        out_audio = numpy.zeros_like(wav)
    else:
        input_wav_path.seek(0)
        # 如有需要自行修改聚类比例cluster_infer_ratio和是否启用杂音过滤功能f0_filter，是否使用自动变调auto_predict_f0
        if raw_infer:
            out_audio, _ = svc_model.infer(speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
              auto_predict_f0=False,
              noice_scale=0.4, f0_filter=False)
            out_audio = out_audio.cpu().numpy()
        else:
            out_audio = svc.process(svc_model, speaker_id, f_pitch_change, input_wav_path)
    # 返回音频
    out_wav_path = io.BytesIO()
    if daw_sample != svc_model.target_sample:
        out_audio = librosa.resample(out_audio,
                                     orig_sr=svc_model.target_sample,
                                     target_sr=daw_sample,
                                     res_type="linear")
    soundfile.write(out_wav_path, out_audio, daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    raw_infer = True
    user_input = input("""
                    实时变声器，默认参数聚类比例0，不使用自动变调，不使用杂音过滤\n
                    如有需要自行开启flask_api.py修改对应内容\n
                    原项目地址：https://github.com/svc-develop-team/so-vits-svc\n
                    vst插件：https://github.com/zhaohui8969/VST_NetProcess-\n
                    技术支持：@串串香火锅\n
                    代码修改、模型训练：@ChrisPreston\n
                    模型使用协议（重要）：\n
                    1.请勿用于商业目的\n
                    2.请勿用于会影响主播本人的行为（比如冒充本人发表争议言论）\n
                    3.请勿用于血腥、暴力、性相关、政治相关内容\n
                    4.不允许二次分发模型与此程序\n
                    5.非个人使用场合请注明模型作者@ChrisPreston\n
                    6.允许用于个人娱乐场景下的游戏语音、直播活动，不得用于低创内容，用于直播前请与我联系\n
                    联系方式：电邮：kameiliduo0825@gmail.com, b站：https://space.bilibili.com/18801308\n
                    カバー株式会社持有最终解释权
                    是否同意上述内容？"""
                       "是y，否n（y/n）")
    if user_input != "y":
        print("正在退出程序")
        exit()
    # 每个模型和config、聚类是唯一对应的
    model_name = "logs/44k/G_198400.pth"
    config_name = "configs/a.json"
    cluster_model_path = "logs/44k/kmeans_10000.pt"
    port = 6847

    # 获取用户输入
    md_replace = input("每个模型和config、聚类是唯一对应的\n"
                       "请输入模型名称：")
    model_name = model_name.replace("G_198400", md_replace)

    cf_replace = input("请输入config名称：")
    config_name = config_name.replace("a", cf_replace)

    cl_replace = input("请输入聚类模型名称（没有就直接回车)：")
    cluster_model_path = cluster_model_path.replace("kmeans_10000", cl_replace)
    # print(cluster_model_path)

    # 获取用户选择的设备
    device_choice = input("请选择设备: 1. cuda 2. cpu ")
    if device_choice == "1":
        device = "cuda"
    else:
        device = "cpu"

    svc_model = Svc(model_name, config_name, device=device, cluster_model_path=cluster_model_path)
    svc = RealTimeVC()
    # 此处与vst插件对应，不建议更改
    app.run(port=port, host="0.0.0.0", debug=False, threaded=False)
