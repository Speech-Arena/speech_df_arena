import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def compute_det_curve(target_scores, nontarget_scores):
    """ frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, score of target (or positive, bonafide) trials
      nontarget_scores: np.array, score of non-target (or negative, spoofed) trials
      
    output
    ------
      frr:         np.array,  false rejection rates measured at multiple thresholds
      far:         np.array,  false acceptance rates measured at multiple thresholds
      thresholds:  np.array,  thresholds used to compute frr and far

    frr, far, thresholds have same shape = len(target_scores) + len(nontarget_scores) + 1
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    # false rejection rates
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    # false acceptance rates
    # print(float(all_scores[indices[0]]) - 0.001)
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    # Thresholds are the sorted scores
    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ eer, eer_threshold = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, score of target (or positive, bonafide) trials
      nontarget_scores: np.array, score of non-target (or negative, spoofed) trials
      
    output
    ------
      eer:              scalar,  value of EER
      eer_threshold:    scalar,  value of threshold corresponding to EER
    """

    frr, far, thresholds = compute_det_curve(
        np.array(target_scores).astype(np.longdouble),
        np.array(nontarget_scores).astype(np.longdouble),
    )
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_f1_accuracy(target_scores, nontarget_scores, threshold):
    """
    Computes F1-score and accuracy based on a given threshold.

    Args:
        target_scores (np.array): Scores for target (bonafide) trials.
        nontarget_scores (np.array): Scores for non-target (spoofed) trials.
        threshold (float): Decision threshold to classify scores.

    Returns:
        f1 (float): F1-score.
        accuracy (float): Accuracy.
    """
    # Assign labels based on threshold
    y_true = np.concatenate([np.ones_like(target_scores), np.zeros_like(nontarget_scores)])
    y_pred = np.concatenate([target_scores >= threshold, nontarget_scores >= threshold])

    # Compute F1-score and accuracy
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return f1, accuracy

def compute_metrics(score_file, protocol_file):
    """
    Computes EER, F1-score, and accuracy from score and protocol files.

    Args:
        score_file (str): Path to the score file.
        protocol_file (str): Path to the protocol file.

    Returns:
        dict: EER, threshold, F1-score, and accuracy.
    """
    # Load protocol and scores
    protocol_df = pd.read_csv(
            protocol_file,
            names=["file_name", "label"],
            index_col="file_name",
            header=0
        ) 
      
    score_df = pd.read_csv(
        score_file,
        names=["file_name", "cm_score"],
        index_col="file_name",
        sep= " ",
    )

    missing = set(protocol_df.index) - set(protocol_df.index).intersection(set(score_df.index))
    assert len(missing) == 0, f"Missing scores for {len(missing)} files"

    # Merge scores with protocol labels
    merged_pd = protocol_df.join(score_df)

    bonafide_scores = merged_pd.query('label == "bonafide"')["cm_score"].to_numpy()
    spoof_scores = merged_pd.query('label == "spoof"')["cm_score"].to_numpy()

    # Compute EER and threshold
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)

    # Compute F1-score and accuracy
    f1, accuracy = compute_f1_accuracy(bonafide_scores, spoof_scores, threshold)

    return {"EER (%)": float(eer*100), "Threshold": float(threshold), "F1-score": float(f1), "Accuracy (%)": float(accuracy*100)}

def compute_pooled_eer(resultds_dict, TIMESTAMP, PROTOCOL_FILES_DIR):
    pooled_eers = {}
    pooled_f1 = {}
    pooled_acc = {}
    pooled_thr = {}

    res_pooled = {}
    for model in resultds_dict:

        all_scores = []
        all_metadata = []

        for score_file in resultds_dict[model]:
            score_path = f'./scores/scores_{TIMESTAMP}/{model}/{score_file}.txt'
            metadata_path = f'{PROTOCOL_FILES_DIR}/{score_file}.csv'

            # Load and store scores and metadata
            df_scores = pd.read_csv(score_path, sep=' ', names=['file_name', 'cm_score'])
            df_meta = pd.read_csv(metadata_path)

            # Align scores and metadata
            df_scores = df_scores.set_index('file_name')
            df_meta = df_meta.set_index('file_name')
            df_combined = df_meta.join(df_scores, how='inner').reset_index()

            all_scores.append(df_combined)

        # Combine all datasets for that model
        pooled_df = pd.concat(all_scores, ignore_index=True)
        
        # Save to temporary files for reuse in compute_metrics
        pooled_df[['file_name', 'cm_score']].to_csv(f'./scores/scores_{TIMESTAMP}/{model}/pooled_scores.txt', sep=' ', index=False, header=False)
        pooled_df[['file_name', 'label']].to_csv(f'./scores/scores_{TIMESTAMP}/{model}/pooled_metadata.csv', index=False)

        # Use existing compute_metrics function
        pooled_res = compute_metrics(f'./scores/scores_{TIMESTAMP}/{model}/pooled_scores.txt',
                                      f'./scores/scores_{TIMESTAMP}/{model}/pooled_metadata.csv')

        res_pooled[model] = pooled_res
    return res_pooled