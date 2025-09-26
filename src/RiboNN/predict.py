from typing import Dict
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import argparse
from pathlib import Path

from RiboNN.data import RiboNNDataModule
from RiboNN.model import RiboNN
from RiboNN.utils.helpers import extract_config


def predict_using_models_trained_in_one_fold(
    run_df: pd.DataFrame,
    config: Dict,
    dm: pl.LightningDataModule,
    top_k_models_to_use: int = 5,
) -> pd.DataFrame:
    """Make predictions averaged across top_k_models listed in mlflow run_df.
    """
    # TE names
    predicted_columns = "TE_108T,TE_12T,TE_A2780,TE_A549,TE_BJ,TE_BRx.142,TE_C643,TE_CRL.1634,TE_Calu.3,TE_Cybrid_Cells,TE_H1.hESC,TE_H1933,TE_H9.hESC,TE_HAP.1,TE_HCC_tumor,TE_HCC_adjancent_normal,TE_HCT116,TE_HEK293,TE_HEK293T,TE_HMECs,TE_HSB2,TE_HSPCs,TE_HeLa,TE_HeLa_S3,TE_HepG2,TE_Huh.7.5,TE_Huh7,TE_K562,TE_Kidney_normal_tissue,TE_LCL,TE_LuCaP.PDX,TE_MCF10A,TE_MCF10A.ER.Src,TE_MCF7,TE_MD55A3,TE_MDA.MB.231,TE_MM1.S,TE_MOLM.13,TE_Molt.3,TE_Mutu,TE_OSCC,TE_PANC1,TE_PATU.8902,TE_PC3,TE_PC9,TE_Primary_CD4._T.cells,TE_Primary_human_bronchial_epithelial_cells,TE_RD.CCL.136,TE_RPE.1,TE_SH.SY5Y,TE_SUM159PT,TE_SW480TetOnAPC,TE_T47D,TE_THP.1,TE_U.251,TE_U.343,TE_U2392,TE_U2OS,TE_Vero_6,TE_WI38,TE_WM902B,TE_WTC.11,TE_ZR75.1,TE_cardiac_fibroblasts,TE_ccRCC,TE_early_neurons,TE_fibroblast,TE_hESC,TE_human_brain_tumor,TE_iPSC.differentiated_dopamine_neurons,TE_megakaryocytes,TE_muscle_tissue,TE_neuronal_precursor_cells,TE_neurons,TE_normal_brain_tissue,TE_normal_prostate,TE_primary_macrophages,TE_skeletal_muscle"
    predicted_columns = predicted_columns.replace("TE_", "predicted_TE_").split(",")

    # Filter run_df to keep the top k models ranked by validation R2
    run_df = run_df.sort_values("metrics.val_r2", ascending=False).head(
        top_k_models_to_use
    )

    # Iterate over the models to make predictions
    predictions = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for run_id in run_df.run_id:
        # Create a new model
        model = RiboNN(**config)

        # Load the state dict
        local_state_dict_path = f"models/{run_id}/state_dict.pth"
        model.load_state_dict(torch.load(local_state_dict_path))
        model.to(device)
        model.eval()

        with torch.no_grad():
            batched_predictions = [model(batch.to(device)) for batch in dm.predict_dataloader()]

        if isinstance(batched_predictions, list):
            batched_predictions = torch.cat(batched_predictions, dim=0)

        predictions.append(batched_predictions.cpu().numpy())

    mean_prediction = np.stack(predictions, axis=-1).mean(axis=-1)

    df = pd.DataFrame(mean_prediction, columns=predicted_columns)

    df = pd.concat([dm.df, df], axis=1)

    return df


def predict_using_nested_cross_validation_models(
    input_path: str,
    run_df: str,
    top_k_models_to_use: int = 5,
    batch_size: int = 1024,
    num_workers: int = 4,
) -> pd.DataFrame:

    # Create data module
    config = extract_config(run_df, run_df.run_id[0])
    config["max_utr5_len"] = 1_381  # used when training the model
    config["max_cds_utr3_len"] = 11_937  # used when training the model
    config["tx_info_path"] = input_path
    config["num_workers"] = num_workers
    config["test_batch_size"] = batch_size
    config["remove_extreme_txs"] = False
    config["target_column_pattern"] = None
    dm = RiboNNDataModule(config)

    all_prediction_dfs = []
    for test_fold in np.sort(run_df["params.test_fold"].unique()):
        test_fold_str = str(test_fold)
        sub_run_df = run_df.query(
            "`params.test_fold` == @test_fold_str or `params.test_fold` == @test_fold"
        ).reset_index(drop=True)
        prediction_df = predict_using_models_trained_in_one_fold(
            sub_run_df, config, dm, top_k_models_to_use
        )
        prediction_df["fold"] = int(test_fold)

        all_prediction_dfs.append(prediction_df)

    all_predictions = pd.concat(all_prediction_dfs, axis=0, ignore_index=True)

    return all_predictions

def predict(input_file: str, output_file: str, models_dir: str, batch_size: int = 32, num_workers: int = 4, top_k_models: int = 5):

    # Create output directory if it does not exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Check if directory with models exists
    model_path = Path(models_dir).resolve()
    if not model_path.exists():
        raise ValueError(f"Directory {model_path} does not exist")

    run_df = pd.read_csv(f"{model_path}/runs.csv")
    predictions = predict_using_nested_cross_validation_models(
        input_file, 
        run_df, 
        top_k_models,
        batch_size=batch_size,
        num_workers=num_workers)

    # Calculate the mean tissue/cell-specific TE across the test folds
    columns_to_aggregate = [col for col in predictions.columns if col.startswith("predicted_")]
    predicted_TE = predictions.groupby(
        ["tx_id", "utr5_sequence", "cds_sequence", "utr3_sequence"],
        as_index=False)[columns_to_aggregate].agg('mean')
    
    # Calculate the mean TE across tissue/cell types
    predicted_TE["mean_predicted_TE"] = predicted_TE[columns_to_aggregate].mean(axis=1)

    # Write predictions to disk
    predicted_TE.to_csv(output_file, index=False, sep="\t")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, help="Path to the input file (tab delimited)", required=True)
    parser.add_argument("--output_file", type=str, help="Path to the output file (tab delimited)", required=True)
    parser.add_argument("--models_dir", type=str, help="Path to the models directory", required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--top_k_models", type=int, default=5, help="Top K models to use for prediction")

    args = parser.parse_args()

    predict(
        args.input_file,
        args.output_file,
        args.models_dir,
        args.batch_size,
        args.num_workers,
        args.top_k_models
    )