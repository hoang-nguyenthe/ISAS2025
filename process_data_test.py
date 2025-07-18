
import os, logging, gc
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from itertools import chain
FPS          = 30
DELTA_T      = 1 / FPS
INPUT_FILE   = 'test data_keypoint.csv'  # hoặc tên file test của bạn
OUTPUT_FILE  = 'features_test.csv'
WINDOW_SIZE  = 45          # frames ≈ 1.5 s

# ------------- MAP -------------
KEYPOINT_COLS_DICT = {
    'rh': ['right_wrist_x','right_wrist_y'],
    'lh': ['left_wrist_x','left_wrist_y'],
    're': ['right_elbow_x','right_elbow_y'],
    'le': ['left_elbow_x','left_elbow_y'],
    'r_shoulder': ['right_shoulder_x','right_shoulder_y'],
    'l_shoulder': ['left_shoulder_x','left_shoulder_y'],
    'r_hip': ['right_hip_x','right_hip_y'],
    'l_hip': ['left_hip_x','left_hip_y'],
    'r_ear': ['right_ear_x','right_ear_y'],
    'l_ear': ['left_ear_x','left_ear_y'],
    'nose': ['nose_x','nose_y'],
    'r_knee': ['right_knee_x','right_knee_y'],
    'l_knee': ['left_knee_x','left_knee_y'],
    'r_ankle': ['right_ankle_x','right_ankle_y'],
    'l_ankle': ['left_ankle_x','left_ankle_y'],
}
KEYPOINT_FLAT   = list(chain.from_iterable(KEYPOINT_COLS_DICT.values()))
EXPECTED_JOINTS = list(KEYPOINT_COLS_DICT.keys())

# ------------- HELPERS -------------
def dominant_frequency(x, fs=FPS):
    n = len(x)
    if n < 2: return 0.0
    yf = rfft(x - x.mean()); mag = np.abs(yf); mag[0] = 0
    return 0.0 if mag.max() == 0 else rfftfreq(n, 1/fs)[mag.argmax()]

def zero_cross_rate(x):
    n = len(x)
    return 0.0 if n < 2 else ((x[:-1] * x[1:]) < 0).sum() / (n - 1)

def calc_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    norm   = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    norm[norm == 0] = 1e-6
    cosang = np.clip((v1 * v2).sum(axis=1) / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def horiz_angle(vec):
    dot = vec[:, 0];  mag = np.linalg.norm(vec, axis=1)
    cosang = dot / (mag + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def movement_regularity(x):
    peaks, _ = find_peaks(x)
    return 0.0 if len(peaks) < 2 else np.std(np.diff(peaks))

# ------------- FEATURES CHO 1 SUBJECT -------------
def build_features_one(df_sub):
    idx   = df_sub.index
    feat  = {}
    cache = {k: df_sub[KEYPOINT_COLS_DICT[k]].values for k in EXPECTED_JOINTS}

    # -- Speed / Acc / Jerk (‖·‖₂, ×FPS) --
    for name in ['rh', 'lh', 'nose']:
        vec   = cache[name]
        vel   = np.diff(vec, axis=0, prepend=vec[:1]) * FPS
        speed = np.linalg.norm(vel, axis=1)
        acc   = np.diff(vel, axis=0, prepend=vel[:1]) * FPS
        acc_m = np.linalg.norm(acc, axis=1)

        feat[f'{name}_speed']        = speed
        feat[f'{name}_acceleration'] = acc_m

        if name in ['rh', 'lh']:
            jerk = np.diff(acc, axis=0, prepend=acc[:1]) * FPS
            feat[f'{name}_jerk'] = np.linalg.norm(jerk, axis=1)

    # Relative tay–mũi
    feat['relative_speed_rh_nose'] = feat['rh_speed'] - feat['nose_speed']
    feat['relative_speed_lh_nose'] = feat['lh_speed'] - feat['nose_speed']
    feat['relative_acceleration_rh_nose'] = feat['rh_acceleration'] - feat['nose_acceleration']
    feat['relative_acceleration_lh_nose'] = feat['lh_acceleration'] - feat['nose_acceleration']

    # -- Elbow --
    for side, sh, el, hd in [('right','r_shoulder','re','rh'),
                             ('left', 'l_shoulder','le','lh')]:
        ang = calc_angle(cache[sh], cache[el], cache[hd])
        feat[f'{side}_elbow_angle']            = ang
        feat[f'{side}_elbow_angular_velocity'] = np.diff(ang, prepend=ang[0]) * FPS

    # -- Simple distances (L2) --
    feat['rh_ear_distance']  = np.linalg.norm(cache['rh'] - cache['r_ear'], axis=1)
    feat['lh_ear_distance']  = np.linalg.norm(cache['lh'] - cache['l_ear'], axis=1)
    feat['rh_nose_distance'] = np.linalg.norm(cache['rh'] - cache['nose'], axis=1)
    feat['lh_nose_distance'] = np.linalg.norm(cache['lh'] - cache['nose'], axis=1)
    feat['distance_hand_nose_diff'] = np.abs(feat['rh_nose_distance'] - feat['lh_nose_distance'])
    feat['speed_asymmetry'] = np.abs(feat['rh_speed'] - feat['lh_speed'])
    feat['rh_y_vs_shoulder_y'] = df_sub['right_wrist_y'] - df_sub['right_shoulder_y']
    feat['lh_y_vs_shoulder_y'] = df_sub['left_wrist_y']  - df_sub['left_shoulder_y']
    feat['hands_distance']     = np.linalg.norm(cache['rh'] - cache['lh'], axis=1)
    feat['speed_diff_hands']   = feat['rh_speed'] - feat['lh_speed']

    # body-centre speed
    body_c = (cache['r_hip'] + cache['l_hip']) / 2
    body_speed = np.linalg.norm(np.diff(body_c, axis=0, prepend=body_c[:1]) * FPS, axis=1)
    feat['body_center_speed'] = body_speed

    # ========== Pandas rolling (C-optimized) ==========
    df_feat = pd.DataFrame(feat, index=idx)  # start assembling

    # nose_y std window
    df_feat['nose_pos_y_std_window'] = df_sub['nose_y'].rolling(WINDOW_SIZE, min_periods=1).std()

    # torso vertical angle + mean window
    vert = np.tile(np.array([0, -1]), (len(df_sub), 1))
    torso_vec = cache['nose'] - body_c
    torso_angle = calc_angle(vert, np.zeros_like(torso_vec), torso_vec)
    df_feat['torso_vertical_angle'] = torso_angle
    df_feat['torso_vertical_angle_window'] = (
        pd.Series(torso_angle, index=idx)
          .rolling(WINDOW_SIZE, min_periods=1).mean()
    )

    # wrist pos std
    for short,long_ in [('r','right'), ('l','left')]:
        for col in ['x','y']:
            df_feat[f'{short}h_pos_std_{col}_window'] = (
                df_sub[f'{long_}_wrist_{col}']
                  .rolling(WINDOW_SIZE, min_periods=1).std()
            )

    # speed & distance rolling stats + ZCR trên Δspeed
    for hand in ['rh', 'lh']:
        s = pd.Series(df_feat[f'{hand}_speed'], index=idx)
        d = pd.Series(df_feat[f'{hand}_nose_distance'], index=idx)
        df_feat[f'{hand}_speed_mean_window'] = s.rolling(WINDOW_SIZE,1).mean()
        df_feat[f'{hand}_speed_std_window']  = s.rolling(WINDOW_SIZE,1).std()
        df_feat[f'{hand}_speed_max_window']  = s.rolling(WINDOW_SIZE,1).max()
        df_feat[f'{hand}_nose_dist_mean_window'] = d.rolling(WINDOW_SIZE,1).mean()
        df_feat[f'{hand}_nose_dist_std_window']  = d.rolling(WINDOW_SIZE,1).std()

        speed_change = s.diff().fillna(0)
        df_feat[f'{hand}_speed_change_zcr_window'] = (
            speed_change.rolling(WINDOW_SIZE,1)
                        .apply(zero_cross_rate, raw=True)
        )
        df_feat[f'{hand}_nose_dist_dom_freq'] = (
            d.rolling(WINDOW_SIZE,1)
              .apply(dominant_frequency, raw=True)
        )

    # legs & shoulders
    for side in ['r','l']:
        knee_ang = calc_angle(cache[f'{side}_hip'], cache[f'{side}_knee'], cache[f'{side}_ankle'])
        df_feat[f'{side}_knee_angle'] = knee_ang
        df_feat[f'{side}_knee_angular_velocity'] = np.diff(knee_ang, prepend=knee_ang[0]) * FPS
        ank_speed = np.linalg.norm(np.diff(cache[f'{side}_ankle'], axis=0,
                                           prepend=cache[f'{side}_ankle'][:1]) * FPS, axis=1)
        df_feat[f'{side}_ankle_speed'] = ank_speed
        df_feat[f'{side}_shoulder_hip_dist'] = np.linalg.norm(
            cache[f'{side}_shoulder'] - cache[f'{side}_hip'], axis=1)

    df_feat['shoulders_line_angle'] = horiz_angle(cache['r_shoulder'] - cache['l_shoulder'])
    df_feat['rh_above_shoulder'] = (cache['rh'][:,1] < cache['r_shoulder'][:,1]).astype(int)
    df_feat['lh_above_shoulder'] = (cache['lh'][:,1] < cache['l_shoulder'][:,1]).astype(int)

    # regularity & acc dominant freq
    for hand in ['rh','lh']:
        acc  = pd.Series(df_feat[f'{hand}_acceleration'], index=idx)
        dist = pd.Series(df_feat[f'{hand}_nose_distance'], index=idx)
        df_feat[f'{hand}_acc_dom_freq'] = acc.rolling(WINDOW_SIZE,1)\
                                            .apply(dominant_frequency, raw=True)
        df_feat[f'{hand}_nose_dist_regularity'] = dist.rolling(WINDOW_SIZE,1)\
                                                    .apply(movement_regularity, raw=True)

    return df_feat.fillna(0)

# ------------- MAIN -------------
def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    logging.info('🚀 Extracting features from test data…')

    if not os.path.exists(INPUT_FILE):
        logging.error(f"❌ Không tìm thấy file {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # Safe interpolation
    df[KEYPOINT_FLAT] = (df[KEYPOINT_FLAT]
                           .interpolate('linear', limit_direction='both')
                           .ffill()
                           .bfill()
                           .fillna(0))

    features = build_features_one(df)
    df_out = pd.concat([df[['frame_id']], features], axis=1)
    df_out.to_csv(OUTPUT_FILE, index=False)

    logging.info(f'✅ Features saved → {OUTPUT_FILE}')

if __name__ == '__main__':
    main()
