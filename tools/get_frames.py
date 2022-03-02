import json
import logging
import os
import shutil
import sys
from multiprocessing.dummy import Pool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.bells import tqdm

from geoscreens.utils import get_indices_to_sample, load_json, save_json, timeit_context


def get_frames_generator_decord(config: DictConfig, video_path: Union[str, Path]):
    vr = VideoReader(str(video_path), ctx=cpu())
    sample_indices = get_indices_to_sample(config, len(vr), vr.get_avg_fps())
    print(
        f"num_frames: {len(vr):,}, num_to_sample: {len(sample_indices):,}, fps: {vr.get_avg_fps()}"
    )
    for sample_idx in tqdm(
        range(len(sample_indices)),
        total=len(sample_indices),
        disable=config.get("disable_progress_bar", False),
    ):
        frame_idx = sample_indices[sample_idx]
        if config.fast_debug and sample_idx >= config.debug_max_frames:
            break
        frame = vr[frame_idx]
        seconds = round(frame_idx / vr.get_avg_fps(), 2)
        yield (frame_idx, seconds, frame)


@timeit_context("extract_frames")
def extract_frames(config: DictConfig, video_path: Path, get_frames_fn: Callable):
    frames_path = Path(config.video_frames_path) / video_path.stem
    if frames_path.exists():
        return (video_path, True, None)
    frames_path.mkdir(exist_ok=True, parents=True)
    try:
        print("Saving frames to: ", frames_path)
        for frame_idx, seconds, frame in get_frames_fn(config, video_path):
            frame_out_path = frames_path / f"frame_{frame_idx:08}-{seconds:010.3f}s.jpg"
            if not frame_out_path.exists():
                cv2.imwrite(str(frame_out_path), cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))
        return (video_path, True, None)
    except Exception as ex:
        print(f"Failed: {video_path}, error: {str(ex)}")
        shutil.rmtree(str(frames_path))
        return (video_path, False, str(ex))


def extract_frames_fake(config: DictConfig, video_path: Path, get_frames_fn: Callable):
    try:
        frames_path = Path(config.video_frames_path) / video_path.stem
        frames_path.mkdir(exist_ok=True, parents=True)
        print("Saving frames to: ", frames_path)
        if "pF9OA332DPk" in str(video_path):
            raise Exception("Fake error")
        return (video_path, True, None)
    except Exception as ex:
        return (video_path, False, str(ex))


def save_frames_metadata(config: DictConfig, files):
    frame_info = {}
    if (Path(config.video_frames_path) / "frame_meta_001.json").exists():
        frame_info = load_json(Path(config.video_frames_path) / "frame_meta_001.json")
    for file in tqdm(files):
        video_id = file.stem
        vr = VideoReader(str(file), ctx=cpu(0))
        sample_indices = get_indices_to_sample(config, len(vr), vr.get_avg_fps())
        frame_info[video_id] = {
            "video_id": video_id,
            "total_frames": len(vr),
            "video_fps": vr.get_avg_fps(),
            "frame_sample_rate_fps": 4.0,
            "num_frames_sampled": len(sample_indices),
        }
    save_json(
        Path(config.video_frames_path) / "frame_meta_002.json",
        frame_info,
    )


def process_videos_muli_cpu(config: DictConfig):
    # fmt: off
    id_list = set([
        "8Bw7td5T49U", "tjTu5QhQgtg", "78xNkZqhB14", "5HueHB6D85g", "TZUU67D8eD4", "5vM8Vn8dzRc", "3e4p2WueJnk", "RoAIPgG1H-I", "D4POQX3geEs", "Fljin_26vug", "8YsakLYpA6I", "qHdMPFeIWf8", "X68B0Gpbbzs", "5DhfpT_BK14", "Xvxoq8uR3Zs", "uOpwZPtkEsk", "vtEQf5SeT8A", "tSwfT6dSwMg", "dO7TdYgtAWg", "RLjrxQJOubY", "mg_sgJJPNqM", "t6K0TXA4FT8", "78lriIFZvVw", "uN4hVUzQC5c", "x9mNJalP73w", "yDXcweSID4c", "5bQFKz_hAD8", "7l1elyK6smk", "8JvPC6mPjCE", "8EB6rbK6KIk", "hYzp8aT7Sqs", "AFTNudiAtWs", "yfNYtA__bJc", "XKIHkUfrTv0", "8TIlYg31Ys8", "7A6OIVDvNCM", "7Cb79FsTSbQ", "H2h8V2zWbNo", "G8muDcrX2vQ", "nr5cEg5dpN0", "k_59zOaQMYI", "ajWkg4k03UY", "XaR8S25aK-0", "EXwUtNeLUio", "FJjsv3hwGx8", "RtDIzWgaHsA", "zO8FOrHRCaE", "7yyMBkkFnl4", "_jRSHbYfCiI", "iS4dwvTr6Aw", "cgXjROAaR14", "Gx1eggPvvz8", "nWNVMm5Cjjc", "0YVorrXo3z0", "fyVctCdaycI", "V8RN33XDge0", "eF443S8Svso", "_w3ymH9Z4lg", "fALNGSKcgaY", "8INPrVUdwbY", "OIbJpKWlTv8", "7sbKs551sSw", "ygJYQcrPwWA", "iREpxVxkVy8", "7ZmbVYKuhJQ", "b56-pFJsYpo", "UNFGeLmSC8c", "yBITEq7yi7w", "z3MEEDh_VJ8", "C920wI0_lTU", "2HW4sUPH-SM", "9H44X63GrUY", "a-jofvbOEog", "ooDLsWuAPRI", "y93i-jEMTyE", "9hdrEQ2M2JQ", "-YqLmQhi2Mw", "5Tnf_wT3oTs", "pwohUNpbLgc", "SynBqjTP1WA", "S0x1s4d5VKY", "RmtlSq159xY", "bvaP9AsDyhM", "lIOFDYsjjO8", "SGdJ1-m0Il8", "Hsd15piApiM", "Hy9DB7WKGhA", "SlJnwQRThiM", "f9REixx1Cjw", "MCu_KujLtZc", "Y-UOwHtvBeo", "ID79S40DMGw", "V1I6qZ-1mo8", "97L6Imlixi8", "Ja5eSUlfrkY", "jbl8tS3LMdM", "TYR_miC66Kk", "hE_kQAVPe9Q", "Df1_jpBKCOI", "1LoeXjd0JwE", "7UXkRbjQPWU", "yYIpaVMG5uY", "T1SHlddMb1w", "TP_5VNhANkE", "n58XMSpPSmQ", "nKztCxH2QoQ", "sHCJccXO-XM", "s2j0LJMz2c8", "jZiZGE5Yw7M", "2RLc782IbG4", "y7o54MXyAmo", "6mfZIQJhzzw", "f4HYBPVfE5U", "hql2k-qRGQA", "XPN6ta4QP1s", "SF56rvgw9oU", "18uwUloxF70", "rBFbCUgtuTE", "XQghH7D9ivI", "YYpk9QuEb4Q", "uAAxSVBG3Ao", "SOQn-tAEqoA", "LYQqQR1-O8k", "9JhJLztPrcY", "-cpjdkdjSe4", "Dtwik5ME6k0", "op4ZXU65IEg", "rv0ExMrsoCQ", "ou6AXRCpTP0", "67h7PRS4DZQ", "8jQ7knB1dDw", "NB5CLD4_m6Y", "ttkayJHBKvM", "0Mh-3dij_yY", "EAcgPhHIz9k", "drAmJ8r8_UI", "NIr1XF0doag", "C085J8_mOFo", "SSiM_QcHUrw", "iG39LbO36OU", "4fpfMCbBZrY", "oPwC9_RkA6U", "xmzgSW9_qQQ", "J5zYNTUdJ7E", "f8Jhio30FcE", "lNAZnqZIe7Q", "Gj6Uf71DPjg", "Uq8-Mb6NuwA", "YMddopmzhaw", "y5o_LqLGTFk", "5PLdPga2l7A", "VFIvAelnjJk", "yZ8ZFqGGHEY", "9Y0iWInV50c", "sKaXS4L_LBE", "J4KwYAUjvps", "jTmdNqzyqoA", "qkgnZOiPhko", "6x7Vjf0eDZU", "dGv87y7orMk", "p8fWooyDeeg", "fGn-Yq7Pf0I", "V-8TDXRUL98", "k8VVUzrKsao", "jsiFQW7YhyU", "YpqCkIfj1kQ", "dY1RXh-43q4", "5ng7eZcmf08", "5Nyey2vqZjc", "3FOClyWnFLo", "_x5vekFIPwM", "bcUskY_-eaI", "frABgE4nvZE", "1gdystGUcWk", "N5PHW5fID5M", "cf1kiPYSQX0", "vK6YBEhT_a0", "j8YGygpejmM", "XzLmdfuT-aw", "Y09e38mJwxg", "BAHGhuTg0uc", "i7RbqNUpkzM", "bseiyQBfDFc", "qrppRbnm0Hs", "jSBbDebSEH8", "aPtUdHUZ3U4", "D-JVd1wiGmM", "FobVijiwcxo", "qQMeHkwP8hg", "tkrwURdzQD4", "erJJ6_dvqXQ", "S8Drmz7kb2Q", "nComDx3Hksk", "iSgTY8VsTf4", "M1ZCdTUpOMk", "KvP2xzlLuVQ", "_Yxhg-Ng6W4", "QAXV5-eUHVI", "UNJO7v2JqrQ", "oxGTI4ifaUI", "fvTmJro2lJs", "ucfvRONvrv4", "7Kov_ocesos", "KA3r-gF1ub8", "WYqh9IOP1ac", "RfGf4Sfi5eo", "dRG76uV8Gh8", "v-GbQnjx9qo", "I8aSkup6v5Y", "zg2Fsd_AQBY", "ByyXpvS5SsM", "L2x3gfC8JKk", "VeV2_pBKwyw", "fitpQnwDSF8", "MVA_CfddM7Q", "0GUAEM7yZfs", "NY3YDQvI1Ic", "ogJnHIuT8Yc", "LR2NyzmxUhU", "VNfQu_7ewMc", "71j61uq6dT0", "--0Kbpo9DtE", "N57v3XC_KgU", "7AAjhFb6vW4", "QeG3JIDj9X0", "eFlRk5-wfqE", "PhwZ9hgWbws", "BktA9fzbBFA", "8gW85SbDGms", "905EaxHE8uM", "2ZjvH8UJmMI", "Pz0sOLzEEc8", "fsHd9Dxb6dg", "jfvF7yBlUzw", "4h-GEDFgbyk", "9N5ehFTyiEA", "o0-B7oMrfYg", "dnw5qOqcUuc", "e7RSZvTrfxM", "S_CkGJ_2NRc", "auNws2QRR-I", "POJmNWuuWoU", "IcZkMPfJt9g", "9jU8kH2W4fw", "5mBTu5fSKrY", "3G7zcxVInzk", "G9iraVNE2YY", "PZNVXTPFpDg", "kheQo4Tpndc", "NhmN-Y4Ogt4", "EnLJRP3URAY", "AF9uezxZDeE", "ARWG_rDyo0w", "svHi_vMMz30", "RUAq8ypbS-M", "fgzsmVt8O9Q", "d9UM7dxWmSo", "HXsnKhMSXnA", "PzAXjKD4ZRg", "nurNqjV2BIE", "Vrc9iyOdffs", "HY_cFHStUdE", "7NpvRgadP6I", "1AAV73eCDng", "SB4UMgTRBe4", "IeN5MHlIFYs", "tqny4LpSUiE", "M1YxSawhLRM", "6dMleMbH1p4", "ycCwEbgAsBA", "28ST7y3V0Ts", "-IEDd8F93a4", "h_v9LNa-CJM", "KjCbBXm68t4", "M9EY3IUfc5k", "l9cwjJ8Fq18", "JPTuaNLK8Wo", "b8uR_dfzk1k", "U9B1tkrtRlE", "gkUCMaoMutQ", "15V7WLGkk_k", "k4IrMBjw4Wg", "XD03_-kYYIU", "qtnVQtoVVVc", "Fr7k4Of9MHc", "T2-lmTdq1xI", "5Zxg-TXyOI8", "FwvekZZF5Uc", "WBnhnil7BpE", "NRfetLNMgko", "khG49U9uA0w", "oipCIsg365Q", "jMRAXSUZfFM", "kAaInBtlT20", "1bYnXDko1Y0", "8F_j53zDM7Y", "ZKSH2u8LA3U", "yFUbV6e_gJU", "ri_34j2BrjQ", "vQ_K9yf88FY", "xEU6jN3g9-M", "mJf_LmgFK5Y", "4Q_7tdz1RLk", "NhWMpSodfiM", "wEhDoTH3z1Q", "Il_uQts188g", "6nqaSvpqq_4", "lUf2aUbevKI", "fNCt6HzDr5s", "uUThkhfRgcI", "NBdNAZ4xMHo", "Eww4ke7BRc8", "P-dCi2PGAbk", "kMxyUZjFRRk", "ZCHRMqF8WYo", "WdmLE97ZN3Y", "YWQsrpXutYI", "daYiBw4mumU", "XzBgSMyXKgc", "64Ig7a6W6Ew", "mkx8bU_di1k", "9FVnGjIxaMY", "CdIz5bo7Djk", "SpevCJaBI3c", "im7nCclNTkQ", "KqBtWllS4gI", "0takfE6ONYc", "8jWG2tLeVMw", "fH17ddueJWU", "x4-N7jQxQiw", "2mifm1pLKAA", "1OXB0WADaXg", "d7Traj8zMS8", "hkSU3XytNkA", "gNevBaS-uGA", "IQRT4fr8emQ", "Y9RLRlKbYdE", "6hFg9JyN2xs", "kx_IvEI7ank", "qB2CxfyYgaM", "bJ7gKGvwpY4", "XikyJPOxAz8", "Qzsf3DWibJk", "CWG3sDtKQDA", "QuiSL6Nkamg", "vtV5A5PcyVs", "ZBGy6D9KzpA", "9dw5HDcPPZc", "rWblyP6DWTI", "Zon59ELN9gY", "4PT2cRmp638", "0NZj6fl8oy8", "ca6GfGUTIBc", "5W-cXIqy7Qk", "1NUz6bWIxAk", "huqYMEJ0XAE", "OfLGN8vG8RI", "9S_aexwPTGY", "ZMNakOAS2PQ", "xFWjYDyiVgA", "oxQaoCK5-gw", "wchMLhhTThU", "DSQ0c3e1JIY", "EgiQcOFcKjI", "4gKwjSDe0pU", "rp0XRhlDaxs", "pEww8bI63pc", "ecTrx01X7Ww", "v3_XOVuqs7Q", "S9isSATsFrM", "Lo0Ss_RBBnE", "d7n1MfAE3Q8", "MZZi2H8arKw", "HTf1OrPoVeI", "3-12cTsYLBI", "26fy28wEjUg", "8QW8dm28hKI", "Jwcovn8B8pQ", "_0-N01oCEgM", "WGn2YyLrV74", "NSYJ8htKe3o", "H-gdaQB8YX8", "om0rl5inBb0", "K29Wr5eheZ0", "J__1wxI_PlI", "oN6DOjX2jwo", "jIao4NbL6RA", "Yv4A_RMBTUo", "7KHfnuxMvmM", "R6nxePpOwj4", "87PTLTwFIlM", "7s7h20eWyNQ", "KZr7oiGwvGo", "RWbEIssom0M", "Y-was9RqOPU", "-PVyrGvE_Ug", "q2pxWYFalxY", "GLbz7V42NTs", "t5BP1LMw7aE", "8qoNZIjLsNQ", "VgXRA5Rs2wY", "2BpNX3tDIp8", "RuPkSF1APFc", "itq0JKF_71k", "dDlabjh0d5M", "EVDZvdWO0Jk", "TWJwleci420", "DXmMV9u9Zu0", "XERAdiWQOjE", "vu_tBi3QEXU", "nBTTbo5aARY", "IIE40ZX84uM", "An7mR7syKAs", "RVntDFkxS7g", "PjBgUuSsA_4", "j7kDcRWH8x0", "LQM9dAUmXbc", "qtsbny47mdw", "3vLgiRXB5XA", "ISRYAscisFs", "yBvuC-gJLRk", "sIck3-vlSlA", "d24qTy_a2y8", "YuDMD_sxMFA", "4QQoRFg_afs", "FR7oa6EyJXw", "ZIJSGDK7JZE", "jQGV_kqnur0", "hZWt1PYH3hI", "5P_6PJMGCcM", "VR8fTKKaxkw", "yAy034yMsKM", "s-ujRzUvgv4", "HLDJp3hetEM", "OJXyK9Grzwg", "xdA779d7rf8", "nAGTYX024vs", "I7MYI_EAW18", "0rxcET_eukc", "tpI_5-rp7B4", "RZKXgfP57ao", "1R0jn0mXAhU", "3b4tPRdX2nk", "BGrHmcYSMqY", "_Wcx5qaUc3E", "nOcf_JCjA6E", "3a3oCg4gDlw", "b_-2uGDp8rU", "iA3wkEutnc8", "IYrmjKKnWsM", "8p8uAN8GMPg", "i1-b_LjegYI", "Uu17FqqvBPc", "0i22gBZe0Vs", "7PV1B8NglOw", "Qj4waaw1OzA", "5mMXaSzIH6M", "M_-STREt-B8", "t2cnHOQr0vQ", "5WPyovVixkY", "4Wloj3k9Gwc", "fsbZAcJXN5s", "az27quZLK8Q", "1963PvpQHlU", "xfIGSpWis6w", "xnvIL-m1Fs4", "U6PpJ8gh8g0", "7y2K0aGKzrI", "1COlCH966vk", "bG0JP1lpvc8", "RPELccF-qHs", "VE7JElrhOZk", "Lqce0YTaqvA", "apViju1pJSQ", "HuMA26sgEW8", "GZmvqMK704U", "kpSA5-dCUco", "-RdNktX6pe0", "4woMwBa9SsI", "TIujifYddi0", "URUobiG8DwM", "WhfHiLV10Dw", "NdX-qjViTnw", "EKU8wGUTXsE", "0LQeo_Ith1M", "dS2CRIjAkSc", "gMaG5H1711A", "EB29A-P4y_I", "uKAHruKnWcs", "3zv372sf060", "j51FXUb-x1Q", "Xg_u0FCyZ74", "Hi2USufrGtI", "JF1uIZRMUf4", "JvL8HHi_cbE", "pnr8PcRoDhU", "FZgw9gz3C1I", "0fbtLGFxJEo", "B6QqER1cyZQ", "1Fho_5uqk98", "KdbXMsZCW_s", "LK1kDgONDyA", "sDYDY5KOA0U", "YwYgHxtSUN8", "MUkhskdIzSw", "BZ9W12amjHo", "U2V25cli1rM", "i_LrB8xGX2Q", "bJmP1QtZRdc", "LXT9U8lhwSk", "yjog5dRToKM", "LlUKiEgP-fg", "7zM6-TZBjfY", "5KvWKvuM0Lw", "XWRcLFQ6SLc", "yQB9yV1aqkM", "4XvGmDO9MH4", "9RQUIk1OwAY", "7DEmYlGENng", "9bHgUYMyuus", "xhJhWN07d28", "QpkW-udMIV0", "Ti-CYdMC41I", "3Hff3DyKB5Y", "i2EytxoXbVw", "XP8VPh-Jck4", "6RtuulgTQEg", "S5gpj3dH04U", "zTnOM3pgsmI", "hPeUmmospU4", "P7oI4IXaQvQ", "42IPZRBbeZU", "R221KrCxkFI", "xiX18l1TJz0", "_pbDtZeN3Os", "eZZW9pr_5yA", "6Uin2eIjfAY", "Tr4OGLLjYnk", "ElcEPfop_CA", "1hu9EumykCw", "HQX9yeK4SuE", "RjzCc0kDGMA", "wPtPW4R3Bz8", "x9w0uk7G-8c", "T6G4wmkcZGg", "zOoUR17xnL0", "KQ7WRBhdKDo", "bIvCkvmf3Xs", "S5Ne5eoHxsY", "S6D18JLde4c", "rs2L2j6SE2Q", "L_PfdixvB1c", "G3FKN03gXIQ", "1NZsTHM_yvA", "PNpUEPXZO6c", "A3PGvLiUQeE", "klYiRchHrh0", "IVzbPudw9cw", "4x7zIn-ypMs", "8IuV-rUEJZQ", "1pvl8Xus5Ek", "9ffhZ_LWL40", "7h-TEPgvKas", "Walstb9S_U4", "5LBrotUmVKg", "U1sSt_i48f8", "6_iwi1av_Tg", "T5UQ0Fab3Ak", "JKwK2CdBka8", "81iad4yS8XA", "2ZU44Af4rGA", "XYVIHqjm7hk", "8I4DWSyhTaQ", "zIph871efJ4", "s3MM2Hn0578", "UO2IzVqYoT0", "JW_cY121vMU", "twuXShfe8ZY", "kXQSKQ-Iglo", "JnnvAlCn4-8", "04Rjc8cHKVA", "kPh7dYErCc8", "ylJlT37wcaE", "o5qTx0bKjrs", "V62niQ9D6wQ", "kko4Sug8_Us", "7KL_zitxz0w", "8HCqhfO4ukc", "5J5OiGLKqQM", "k-dT_m1bSuE", "8ytmWvud6-4", "S3MGYVhx2gM", "1s3Ax_KrLs0", "F6quWYzK7TU", "Z3V98L7YL00", "RH9p7jUHTBw", "-C7NJpYc3gw", "hpsYcBqu5gc", "wriKnOjJy6E", "c78FrBan-JQ", "MocC6pni5P8", "DuL3xoy_BAY", "DHhTo7sZrDg", "td1eP9FSL3Q", "1V35b_G7wok", "ThHk_Q1uovY", "VGNEtq4Bw3U", "K4GXuDACK40", "P8o_guQi41s", "YImUOaH7Uwg", "NC07yL_yN20", "hcdEoiWKPns", "T2r-_WM0enQ", "NrqC1YuBCuQ", "mBHXqKnDo6s", "1AmqyFQYSCU", "2he1eFOivTM", "Rnpm3wKt0Ls", "cpeIVVLO5QE", "1apsDpo_cGI", "Sn0XIA2aux4", "9Xr6LqfOIg0", "XvpAfKb67YE", "DsMej1SWfzU", "D7d-Wilesgw", "1hbFSXAZlro", "GtSNM5lM6Ww", "qRW9erqzcUA", "GungykjFga8", "QBJdM-PmiBY", "1AibHUd6TO4", "DJV5YcQ-lnU", "aJc9vJNv4G4", "2kcCNL3lxDw", "2cqIk-0WSdA", "M3BFifwYolI", "rTIYFLh2Y2Y", "mr08FdZmLa0", "DaX4ZuINsgg", "OLkXxaP5RRs", "pvjUGDcegGo", "mCDGajhSzn8", "U73B2rWWz4U", "ut6DwUYicv8", "OKOP1dnLRJE", "9fRE0IuFcZ8", "sA19OnfmFd4", "vnlSCnx5UUg", "2HGOjVDc1mY", "elgUkTn8snE", "OFuGaf8od_Q", "PzGs-GY1DaA", "SsQ4uMnSJRo", "SKxA0OEnCnk", "B511HI1RHuI", "MWR-h25GwmA", "89QEfKzSkzY", "0KJ0qXEVACs", "clB89mi-ZlY", "W82qLlUxuy0", "cKlp2iLl8Ps", "e-E1-hpK4w0", "khkp8TrFytQ", "a-8DyDJvglU", "KPVPU3vl_yI", "83m9ys4kxro", 
        "jsqkwpdurgw", "OD2_TpEG-C0", "YHHAQkgs32M", "XM87FwIenH4", "C1RHyCSkkKM", "ppYi2iGMht4", "TKTZoyKhHco", "19c03AjUyI8", "74jleDoImYc", "k6Cw-X3zDuo", "uCbXqNRiHRY", "HRsqlJmj-hE", "P9SWbIShXc8", "IV4Pb__t6ys", "_EJjfqyaebo", "5mTvphLejLU", "K2XcndcvzSc", "Sgp-xuDo1gs", "PF0_X0ilLNQ", "j-FROoAWAdw", "i7K9f9l05qA", "7UJ6f-iP6Bc", "7nb9W76qdEc", "1bKzHmmoofg", "osTwgzWluVs", "4mUPPVoQ-kI", "XTKWfDyejck", "3nPxLoDdlxY", "nqlxwD5RY94", "XoHvL7T10pM", "1g4aAtsCUCI", "87qBSgbwLq8", "C8l-D_z2p6k", "bbBeyBFvAyQ", "dVFTw5pacsY", "ZDAZvXbmOSs", "bz04LZDf0Oo", "vAtsI9ZA11U", "LtaV054F5Rs", "ldyQwxNCzlo", "AIHJQEtcPqY", "NjriHMSM26k", "pcxL0W0RuTk", "XjzDqqvMEcU", "V4D-N71EvvM", "hVBhp7U0xQQ", "8__DkawMrPo", "alKpLiE-KY4", "4ftLC-J963E", "HlAZm63ooQY", "OQFa34FoVY8", "KSlCyDyda1M", "SAmkg58TybQ", "IyfiCVAeyhg", "__qlp6JpcPc", "Tjxb3UzaduA", "-C2718vV9v4", "hYMPlDM5j80", "hzm0mRZAOrY", "X-3h3RTPoRg", "yDO_FdDeHfs", "IpWokx3CtDg", "1rpQNdmTtvk", "76F1NoCzNLY", "V2H1po51T-A", "A960qcFjXK8", "MaVbMOJg4Mg", "fBfmMCCMXHk", "CYe9uv5G1hw", "Gk-1toFmMZo", "jCJoc-6iqng", "Y8yW_BsZ018", "6FKmF1LDPrw", "1ShNH_1lVsw", "iT1c92P40z8", "DE-6w16AhcQ", "jNKj2MXeah4", "w-aTpX4_vcg", "CC8CCvle0Vc", "bErSxaBHc1U", "BslAPtFdAnE", "Mv8DWoW5Ojs", "DU7SmB9yqpQ", "ZKCPfy5gcKs", "pFZoRi-T9c0", "q5FEts6gKII", "hzjA9gfxMeQ", "TDL3Jk3SaK4", "FGGfQH0XxL0", "zFuNkVjC6hQ", "tOXrATngKOY", "18C7YgxCkIw", "GvTVU0Qslc0", "HKPpZx0E8FY", "no5rJt2Pj-0", "2SzL5VBF_BI", "8LXi_tpkpSg", "DJp6sBH46KI", "y2eFiEO3BdQ", "pP4VQ9JClM4", "Y0BjWF5gWmk", "wvl9ecRASFM", "l6TtoNu1_Jg", "88jRjbWTesc", "GbgHjGPVzao", "5LXv8t_-6_c", "LcIkJQgAOyA", "5PGCL_L81UA", "y2SDYOw0SDU", "o1GEEkLVFsk", "54c2PpV65hU", "ti2co-2tymo", "hdUTzzDEV8g", "RBPYrNfjnHY", "5IyXNi8TbzM", "t9YiBXcb49g", "qi3_M5Udics", "2ZioLZEZTdI", "dEi4H8QFxIg", "WtzVRdkaY7Y", "26VxX9_chSw", "6wrhV6SnfnE", "3xm58Ccps1c", "MKSRbvmbTVo", "NsUjPBgFJGw", "hXYKSf6QO04", "B7_WcWmSZ_Y", "vKWbEE-ttYA", "0J7cQ4FiDCc", "N_OLaiTShLA", "BuZ-ld13p1I", "izjFsRDbt5k", "TVt1GKBZMzc", "1kwoT-9ClV4", "BJ-bEId-7yQ", "JXUWefGu67A", "WD9ARP45p_M", "V6rWoMiKsAw", "WqKqqN54fHo", "WEqf7L2KL4g", "itGph3begOY", "ep8WGVBNq7M", "67SQuXM5J0o", "tuGOZme6Xjg", "fs98qK3Lnpk", "88FFBre9GFA", "D4fiQwrF5vw", "FTcYSsoAw28", "EAoVoLtzovk", "edPiYaSZvx0", "6h0faOkFVpI", "euFnkGaA5IY", "AX5D2Bt__gE", "IwX1wEQsngE", "DkLLhsfa-PU", "R1toABTIp9s", "AXxUyPWLoJA", "sf0ToMiRIvM", "zlsM4XM9_Rk", "4k57dSg8Cws", "iy8jWAIaG0w", "YZxqCnxHPtE", "TJbSoBC1qpc", "FGidXjJl1vA", "JyfpZuVSisU", "01zaYTxpmxk", "-L9-kfYb4_A", "ta4rw6YR8LQ", "7G_CTikQVEM", "PoNt0SfS5jk", "kdmw3fmfg88", "rb4dy7HLH8M", "HKTCO8mh5d8", "zfZ6BxPne4E", "nJPm9WqyTTU", "22v6KBUXEeQ", "rbfjUayf-gk", "wNr9eLPIits", "VQI85nD7h4U", "96Th1-UjSOU", "OSYcvcQn2w8", "hG6rJf0RBnk", "yYEk84kRqPo", "nuqUwvdZ30M", "hEZVNDqid2I", "pMgqa0mOExo", "E-rvEqFNBcs", "4aUHm5A_66w", "obF2AR2e5P8", "uhbki9G10zU", "GjuSYrjqaMs", "_osOExSsNyk", "UPWSMhAF3ME", "I_5u_woAjm4", "qQyXQtLoGyE", "o8qQAjkaXMM", "tCqhl589AdM", "m0e92O4DJKE", "_MBsJDwzn1M", "nyHeQWnm8YA", "kCSNzVDJ_W4", "7vz-osmi6tw", "Mk9x9VZpIi4", "De1Xc6EXj9Q", "UCQg1LJOywc", "HdQjTia26y4", "WMvw_CCYeG0", "utv4vBdgSG0", "liNzvqszWPc", "j4SXWDgDSSE", "ZcFbyUJbP94", "1GehhMoUwnc", "j_UyjWUW-cU", "jjTvJdgmsmc", "OwLe8JNyynw", "el4lgYNq6mY", "9w_En85TFqc", "E0DHszfXnsc", "7-uBrcBKCpE", "iRTAdF_o1-4", "nXx6gUklog8", "wnH6quY_MUE",
    ])
    # fmt: on
    files = sorted(Path(config.videos_path).glob("*.mp4"))
    files = [f for f in files if f.stem not in id_list][:1000]
    print("Num videos: ", len(files))

    num_workers = config.get("num_workers", 16)
    args = ((config, video_path, get_frames_generator_decord) for i, video_path in enumerate(files))
    with Pool(processes=num_workers) as pool:
        result = pool.starmap(extract_frames, args)
        df_results = pd.DataFrame(
            {
                "video_path": [r[0] for r in result],
                "success": [bool(r[1]) for r in result],
                "error": [r[2] for r in result],
            }
        )
        print("")
        print("")
        print("Failed videos:")
        print(df_results[~df_results.astype(bool).success])

    save_frames_metadata(config, files)


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.max_rows", None)
    # Suitable default display for floats
    pd.options.display.float_format = "{:,.2f}".format
    config = DictConfig(
        {
            "frame_sample_rate_fps": 4.0,
            "fast_debug": False,
            "debug_max_frames": 30,
            # "video_frames_path": "/home/gbiamby/proj/geoscreens/data/video_frames",
            # "videos_path": "/home/gbiamby/proj/geoscreens/data/videos",
            "video_frames_path": "/shared/gbiamby/geo/video_frames",
            "videos_path": "/shared/g-luo/geoguessr/videos",
            "num_workers": 16,
            "disable_progress_bar": True,
        }
    )
    process_videos_muli_cpu(config)
