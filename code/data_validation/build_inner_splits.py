
from pathlib import Path
import pandas as pd
import hashlib, json, argparse

SEED = 2026
VERSION = "inner_clean_v2_v1"
STATUS = "PENDING_TEAM_APPROVAL"

LABELS = [
    "SEG_TRAIN", "META_FIT", "CALIBRATION",
    "SEG_VAL", "OUTER_TEST", "NEVER_POLYP_TEST"
]

TARGET = {
    "SEG_TRAIN": 0.70,
    "META_FIT": 0.20,
    "CALIBRATION": 0.10
}

PATIENT_FILE = "patient_splits_clean_cv5_v2_proposal.csv"
IMAGE_FILE = "image_splits_clean_cv5_v2_proposal.csv"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def paths(out):
    out = Path(out)
    return (
        out / "data/splits/inner_clean_v2_v1",
        out / "results/data_splits/inner_clean_v2_v1"
    )

def load(source):
    source = Path(source)
    p = pd.read_csv(source/PATIENT_FILE, dtype=str, keep_default_na=False)
    i = pd.read_csv(source/IMAGE_FILE, dtype=str, keep_default_na=False)
    linkage = pd.read_csv(
        source/"patient_linkage.csv", dtype=str, keep_default_na=False
    )
    with open(source/"validation_summary.json", encoding="utf-8") as f:
        original_audit = json.load(f)
    assert len(p) == 2275
    assert len(i) == 40365
    assert p.groupby("fold").size().eq(455).all()
    assert i.groupby("fold").size().eq(8073).all()
    assert set(p["fold"]) == set("01234")
    assert not p.duplicated(["fold","patient_id"]).any()
    assert not i.duplicated(["fold","image_id"]).any()
    assert set(i["split"]) == set(p["proposed_split"])
    return p, i, linkage, original_audit

def stable_key(fold, gid):
    s = f"{SEED}:{fold}:{gid}".encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def allocate(group_df, fold):
    # Deterministic greedy balancing:
    # Polyp first, group count second, Healthy third.
    names = list(TARGET)
    total_p = max(1, int(group_df["polyp"].sum()))
    total_g = max(1, len(group_df))
    total_h = max(1, int(group_df["healthy"].sum()))

    current = {
        s: {"polyp":0,"groups":0,"healthy":0}
        for s in names
    }

    rows = group_df.to_dict("records")
    rows.sort(key=lambda r: (
        -int(r["polyp"]),
        -int(r["healthy"]),
        stable_key(fold, r["split_group_id"])
    ))

    assignments = {}

    def score(counts):
        v = 0.0
        for s in names:
            t = TARGET[s]
            c = counts[s]
            v += 100 * (c["polyp"]/total_p-t)**2
            v += 4 * (c["groups"]/total_g-t)**2
            v += 1 * (c["healthy"]/total_h-t)**2
        return v

    for row in rows:
        gid = row["split_group_id"]
        best = None

        for s in names:
            candidate = {
                k: dict(v) for k,v in current.items()
            }
            candidate[s]["polyp"] += int(row["polyp"])
            candidate[s]["healthy"] += int(row["healthy"])
            candidate[s]["groups"] += 1
            value = (score(candidate), names.index(s))

            if best is None or value < best[0]:
                best = (value, s)

        chosen = best[1]
        assignments[gid] = chosen
        current[chosen]["polyp"] += int(row["polyp"])
        current[chosen]["healthy"] += int(row["healthy"])
        current[chosen]["groups"] += 1

    return assignments

def build(source, out):
    p, i, linkage, original_audit = load(source)
    data_dir, result_dir = paths(out)
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the accepted Clean V2 outer partitions.
    mapping = {
        "VAL":"SEG_VAL",
        "TEST":"OUTER_TEST",
        "NEVER_POLYP_TEST":"NEVER_POLYP_TEST"
    }

    p = p.copy()
    p["original_split"] = p["proposed_split"]
    p["inner_split"] = p["proposed_split"].map(mapping)

    for fold in sorted(p["fold"].unique()):
        pf = p[(p["fold"]==fold) & (
            p["proposed_split"]=="TRAIN"
        )]
        frame = i[(i["fold"]==fold) & (i["split"]=="TRAIN")]

        groups = frame.groupby("split_group_id").agg(
            polyp=("category",lambda x:int((x=="Polyp").sum())),
            healthy=("category",lambda x:int((x=="Healthy").sum()))
        ).reset_index()

        expected = set(pf["split_group_id"])
        assert set(groups["split_group_id"]) == expected

        selected = allocate(groups, fold)

        mask = (p["fold"]==fold) & (
            p["proposed_split"]=="TRAIN"
        )

        p.loc[mask,"inner_split"] = (
            p.loc[mask,"split_group_id"].map(selected)
        )

    assert p["inner_split"].notna().all()

    patient_cols = [
        "fold","patient_id","split_group_id",
        "original_split","inner_split","cohort",
        "effective_group_cohort"
    ]

    result_p = p[patient_cols].copy()
    result_p["seed"] = str(SEED)
    result_p["protocol_version"] = VERSION
    result_p["status"] = STATUS
    result_p = result_p.sort_values(
        ["fold","patient_id"]
    ).reset_index(drop=True)

    result_i = i.merge(
        result_p[["fold","patient_id","inner_split"]],
        on=["fold","patient_id"],
        validate="many_to_one"
    ).rename(columns={"split":"outer_split"})

    result_i["protocol_version"] = VERSION
    image_cols = [
        "fold","image_id","patient_id","split_group_id",
        "category","image_path","mask_path","sha256",
        "outer_split","inner_split","protocol_version"
    ]
    result_i = result_i[image_cols].sort_values(
        ["fold","image_id"]
    ).reset_index(drop=True)

    result_p.to_csv(
        data_dir/"patient_inner_splits_cv5_candidate.csv",
        index=False,encoding="utf-8-sig"
    )
    result_i.to_csv(
        data_dir/"image_inner_splits_cv5_candidate.csv",
        index=False,encoding="utf-8-sig"
    )

    return result_p, result_i

def validate(source, out):
    original_p, original_i, linkage, original_audit = load(source)
    data_dir, result_dir = paths(out)

    pfile = data_dir/"patient_inner_splits_cv5_candidate.csv"
    ifile = data_dir/"image_inner_splits_cv5_candidate.csv"

    p = pd.read_csv(pfile,dtype=str,keep_default_na=False)
    i = pd.read_csv(ifile,dtype=str,keep_default_na=False)

    assert len(p)==2275
    assert len(i)==40365
    assert set(p["inner_split"])==set(LABELS)
    assert not p.duplicated(["fold","patient_id"]).any()
    assert not i.duplicated(["fold","image_id"]).any()
    assert p.groupby("fold").size().eq(455).all()
    assert i.groupby("fold").size().eq(8073).all()

    group_conflicts = p.groupby(
        ["fold","split_group_id"]
    )["inner_split"].nunique()
    assert group_conflicts.max()==1

    old = original_p[
        ["fold","patient_id","proposed_split"]
    ]
    pp = p.merge(old,on=["fold","patient_id"],
                 validate="one_to_one")
    assert len(pp)==2275
    assert (pp["original_split"]==pp["proposed_split"]).all()

    allowed = {
        "TRAIN":{"SEG_TRAIN","META_FIT","CALIBRATION"},
        "VAL":{"SEG_VAL"},
        "TEST":{"OUTER_TEST"},
        "NEVER_POLYP_TEST":{"NEVER_POLYP_TEST"}
    }
    for old_label, choices in allowed.items():
        sub = pp[pp["original_split"]==old_label]
        assert set(sub["inner_split"]).issubset(choices)

    ii = i.merge(
        p[["fold","patient_id","split_group_id","inner_split"]],
        on=["fold","patient_id"],
        suffixes=("_image","_patient"),
        validate="many_to_one"
    )
    assert len(ii)==40365
    assert (
        ii["split_group_id_image"]==
        ii["split_group_id_patient"]
    ).all()
    assert (
        ii["inner_split_image"]==
        ii["inner_split_patient"]
    ).all()

    old_i = original_i[
        ["fold","image_id","split","sha256","category"]
    ]
    check_i = i.merge(
        old_i,on=["fold","image_id"],
        validate="one_to_one"
    )
    assert len(check_i)==40365
    assert (check_i["outer_split"]==check_i["split"]).all()
    assert (check_i["sha256_x"]==check_i["sha256_y"]).all()
    assert (check_i["category_x"]==check_i["category_y"]).all()

    assert i.groupby("fold")["sha256"].nunique().eq(8073).all()
    assert i[i["category"]=="Polyp"].groupby(
        "fold"
    ).size().eq(430).all()

    for a,b in [("Patient35","Patient36"),
                ("Patient366","Patient386")]:
        x = p[p["patient_id"]==a].set_index("fold")
        y = p[p["patient_id"]==b].set_index("fold")
        assert (x["inner_split"]==y["inner_split"]).all()

    # No historically polyp-linked group in never-polyp test.
    bad = p[
        (p["inner_split"]=="NEVER_POLYP_TEST")&
        (p["effective_group_cohort"]!="NEVER_POLYP_GROUP")
    ]
    assert len(bad)==0

    # Every original Polyp cohort-linked group is tested once.
    polyp_groups = p[
        p["effective_group_cohort"]=="POLYP_LINKED_GROUP"
    ].drop_duplicates(["fold","split_group_id"])
    tested = polyp_groups.assign(
        tested=polyp_groups["inner_split"]=="OUTER_TEST"
    ).groupby("split_group_id")["tested"].sum()
    assert len(tested)==187
    assert tested.eq(1).all()

    records=[]
    warnings=[]

    for fold in sorted(p["fold"].unique()):
        train_poly = int(len(i[
            (i["fold"]==fold)&
            (i["outer_split"]=="TRAIN")&
            (i["category"]=="Polyp")
        ]))

        for split in LABELS:
            ps = p[(p["fold"]==fold)&
                   (p["inner_split"]==split)]
            ims = i[(i["fold"]==fold)&
                    (i["inner_split"]==split)]
            polyp = int((ims["category"]=="Polyp").sum())
            healthy = int((ims["category"]=="Healthy").sum())

            if split in ("SEG_TRAIN","META_FIT","CALIBRATION"):
                assert polyp>0 and healthy>0, (
                    f"Empty class: fold {fold} {split}"
                )
                if polyp<15:
                    warnings.append(
                        f"Fold {fold}: {split} only {polyp} Polyp images"
                    )

            records.append({
                "fold":fold,
                "inner_split":split,
                "patients":len(ps),
                "split_groups":ps["split_group_id"].nunique(),
                "polyp":polyp,
                "healthy":healthy,
                "polyp_fraction_of_outer_train": (
                    round(polyp/train_poly,5)
                    if split in ("SEG_TRAIN","META_FIT","CALIBRATION")
                    else ""
                )
            })

    stats=pd.DataFrame(records)
    assert len(stats)==30
    assert stats.groupby("fold")["polyp"].sum().eq(430).all()
    assert stats.groupby("fold")["healthy"].sum().eq(7643).all()

    result_dir.mkdir(parents=True,exist_ok=True)
    stats.to_csv(
        result_dir/"fold_stats.csv",
        index=False,encoding="utf-8-sig"
    )

    checksum_inputs = {
        f:sha(Path(source)/f)
        for f in [
            PATIENT_FILE,IMAGE_FILE,
            "patient_linkage.csv","validation_summary.json"
        ]
    }
    checksum_outputs = {
        pfile.name:sha(pfile),
        ifile.name:sha(ifile)
    }

    audit = {
        "status":STATUS,
        "seed":SEED,
        "protocol_version":VERSION,
        "result":"PASS",
        "patient_rows":len(p),
        "image_rows":len(i),
        "linked_group_conflicts":0,
        "patient_image_mismatches":0,
        "outer_partition_changes":0,
        "cross_split_exact_sha_duplicates":0,
        "warnings":warnings,
        "checkpoint_warning":
          "CHECKPOINT_INCOMPATIBLE_WITH_HOLDOUT_META_FIT "
          "if trained on original full TRAIN",
        "input_sha256":checksum_inputs,
        "output_sha256":checksum_outputs
    }

    (result_dir/"inner_split_audit.json").write_text(
        json.dumps(audit,indent=2,ensure_ascii=False),
        encoding="utf-8"
    )

    table = stats.to_markdown(index=False)

    (result_dir/"audit_report.md").write_text(
        "# Inner Clean V2 CV5 Audit\n\n"
        "**Status:** PENDING_TEAM_APPROVAL\n\n"
        "## Assertions\n"
        "- PASS: 2275 patient-fold records\n"
        "- PASS: 40365 image-fold records\n"
        "- PASS: 455 patients and 8073 images per fold\n"
        "- PASS: 430 Polyp images per fold\n"
        "- PASS: Preserve outer TRAIN/VAL/TEST/NEVER partitions\n"
        "- PASS: Patient/group/image mapping consistency\n"
        "- PASS: Linked-patient constraints\n"
        "- PASS: No repeated SHA-256 within folds\n"
        "- PASS: META_FIT and CALIBRATION contain both classes\n\n"
        "## Fold statistics\n\n"+table+"\n\n"
        "## Warnings\n"+
        ("\n".join("- "+w for w in warnings)
         if warnings else "- None from count checks")+
        "\n\n**Not tested here:** Runtime M3 same-group retrieval "
        "exclusion. It must be enforced in retrieval code.\n",
        encoding="utf-8"
    )

    (result_dir/"decision_note.md").write_text(
        "# Inner Split Decision Note\n\n"
        "**Status:** PENDING_TEAM_APPROVAL\n\n"
        "Protocol A, seed 2026; preserve outer partitions. "
        "Allocate only outer TRAIN by split_group_id. "
        "Deterministic greedy objective prioritizes Polyp image "
        "counts (~70/20/10), then group counts and Healthy counts. "
        "Group indivisibility takes precedence over exact quotas.\n\n"
        "## Module policy\n"
        "- M1: train on SEG_TRAIN Polyp, select on SEG_VAL.\n"
        "- M2: mine/fine-tune on SEG_TRAIN only, select on SEG_VAL.\n"
        "- M3: build approved memory on META_FIT; exclude the "
        "query's patient and linked split_group_id in retrieval.\n"
        "- M4: fit on META_FIT using an unseen backbone; "
        "choose thresholds only on CALIBRATION.\n"
        "- OUTER_TEST and NEVER_POLYP_TEST: final evaluation only.\n\n"
        "## Checkpoint compatibility\n"
        "A prior checkpoint trained on the original full TRAIN "
        "cannot be treated as unseen for META_FIT. "
        "Keep old checkpoints; retrain M1/M2 after approval.\n\n"
        "## Fold statistics\n\n"+table+"\n\n"
        "## Remaining approval\n"
        "Team must approve this candidate and the module protocol "
        "before official use or training.\n",
        encoding="utf-8"
    )
    return stats, audit

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["build","validate"])
    parser.add_argument("--inputs",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()

    if args.mode=="build":
        build(args.inputs,args.out)
    else:
        stats,audit=validate(args.inputs,args.out)
        print(stats.to_string(index=False))
        print("VALIDATION: PASS")

if __name__=="__main__":
    main()
