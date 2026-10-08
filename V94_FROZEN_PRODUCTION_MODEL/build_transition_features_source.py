def build_transition_features(df, fit_reference):
    out = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # A. Individual physically meaningful movement features
    # --------------------------------------------------------
    raw_signal_cols = []
    for tag in TRANSITION_TAGS:
        for suffix in ["RT_DELTA", "RT_SLOPE", "END30_DELTA", "END30_SLOPE"]:
            c = f"RTWIN_{tag}_{suffix}" if suffix.startswith("RT_") else f"END30_{tag}_{suffix}"
            if c in df.columns:
                raw_signal_cols.append(c)
                out[f"V93_ABS_{c}"] = pd.to_numeric(df[c], errors="coerce").abs()

    # --------------------------------------------------------
    # B. Recent-vs-RT trajectory displacement
    # --------------------------------------------------------
    shift_cols = []
    for tag in TRANSITION_TAGS:
        rt = f"RTWIN_{tag}_MEAN"
        recent = f"END30_{tag}_MEAN"
        if rt in df.columns and recent in df.columns:
            name = f"V93_STATE_SHIFT_{tag}"
            out[name] = (
                pd.to_numeric(df[recent], errors="coerce")
                - pd.to_numeric(df[rt], errors="coerce")
            )
            shift_cols.append(name)

    # --------------------------------------------------------
    # C. Training-fold-only robust movement scores
    # --------------------------------------------------------
    z_cols = []
    for c in out.columns:
        if c.startswith("V93_ABS_"):
            # Match the same engineered column on the reference frame.
            # Recreate it from the reference source column.
            source = c.replace("V93_ABS_", "", 1)
            if source in fit_reference.columns:
                ref_abs = pd.to_numeric(fit_reference[source], errors="coerce").abs()
            else:
                continue
            med, iqr = robust_center_scale(ref_abs)
            z = robust_z(out[c], med, iqr)
            zname = c.replace("V93_ABS_", "V93_ROBUST_MOVE_")
            out[zname] = z
            z_cols.append(zname)

    for c in shift_cols:
        # Recreate the reference displacement from source tags.
        tag = c.replace("V93_STATE_SHIFT_", "", 1)
        rt = f"RTWIN_{tag}_MEAN"
        recent = f"END30_{tag}_MEAN"
        if rt not in fit_reference.columns or recent not in fit_reference.columns:
            continue
        ref_shift = (
            pd.to_numeric(fit_reference[recent], errors="coerce")
            - pd.to_numeric(fit_reference[rt], errors="coerce")
        )
        med, iqr = robust_center_scale(ref_shift)
        zname = f"V93_ROBUST_STATE_SHIFT_{tag}"
        out[zname] = robust_z(out[c], med, iqr)
        z_cols.append(zname)

    # --------------------------------------------------------
    # D. Composite process-transition intensity
    # --------------------------------------------------------
    if z_cols:
        zmat = out[z_cols].abs()
        out["V93_PROCESS_TRANSITION_SCORE"] = zmat.mean(axis=1, skipna=True)
        out["V93_MAX_TRANSITION_Z"] = zmat.max(axis=1, skipna=True)

        # Number of unusually large movements, with thresholds learned
        # from the chronological training reference only.
        high_count = pd.Series(0.0, index=df.index)
        valid_count = pd.Series(0.0, index=df.index)

        for c in z_cols:
            # z itself is standardized from the training reference;
            # threshold 1.0 means approximately one IQR from center.
            valid = out[c].notna()
            high_count.loc[valid] += (out.loc[valid, c].abs() >= 1.0).astype(float)
            valid_count.loc[valid] += 1.0

        out["V93_N_HIGH_TRANSITIONS"] = high_count
        out["V93_FRACTION_HIGH_TRANSITIONS"] = (
            high_count / valid_count.replace(0, np.nan)
        )

    # --------------------------------------------------------
    # E. Residence-time abnormality
    # --------------------------------------------------------
    rt_cols = [
        c for c in [
            "TOTAL_RT_HR_TRAJ",
            "DRR_RT_HR_TRAJ",
            "PP2_RT_HR_TRAJ",
            "PP1_RT_HR_TRAJ",
            "EST2_RT_HR_TRAJ",
            "EST1_RT_HR_TRAJ",
        ] if c in df.columns and c in fit_reference.columns
    ]

    rt_z_cols = []
    for c in rt_cols:
        ref = pd.to_numeric(fit_reference[c], errors="coerce")
        med, iqr = robust_center_scale(ref)
        zname = f"V93_RT_ABS_Z_{c.replace('_HR_TRAJ','')}"
        out[zname] = robust_z(pd.to_numeric(df[c], errors="coerce"), med, iqr).abs()
        rt_z_cols.append(zname)

    if rt_z_cols:
        out["V93_RT_ANOMALY_SCORE"] = out[rt_z_cols].mean(axis=1, skipna=True)
        out["V93_MAX_RT_ANOMALY"] = out[rt_z_cols].max(axis=1, skipna=True)

    return out.replace([np.inf, -np.inf], np.nan)
