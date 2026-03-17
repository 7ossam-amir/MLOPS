"""Main entry point for GAN training and evaluation."""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent


def _find_kaggle_project_root() -> Optional[Path]:
    """Find project files inside attached Kaggle datasets, if present."""
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None

    for src_dir in input_root.rglob("src"):
        candidate = src_dir.parent
        if (candidate / "config.yaml").exists():
            return candidate
    return None


def _bootstrap_import_path() -> None:
    """Ensure src imports work locally and inside Kaggle script kernels."""
    global PROJECT_ROOT

    local_ready = (PROJECT_ROOT / "src").is_dir() and (PROJECT_ROOT / "config.yaml").exists()
    if local_ready:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        return

    kaggle_root = _find_kaggle_project_root()
    if kaggle_root is not None:
        PROJECT_ROOT = kaggle_root
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_config_path(config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.is_absolute():
        return config_path
    if config_path.exists():
        return config_path.resolve()
    return PROJECT_ROOT / config_path


def _resolve_data_path(configured_path: str) -> Path:
    data_path = Path(configured_path)
    if data_path.is_absolute() and data_path.exists():
        return data_path

    if not data_path.is_absolute():
        cwd_candidate = Path.cwd() / data_path
        if cwd_candidate.exists():
            return cwd_candidate

        root_candidate = PROJECT_ROOT / data_path
        if root_candidate.exists():
            return root_candidate

    input_root = Path("/kaggle/input")
    if input_root.exists():
        default_candidate = input_root / "digit-recognizer" / "train.csv"
        if default_candidate.exists():
            print(f"Using fallback Kaggle dataset path: {default_candidate}")
            return default_candidate

        for found in input_root.rglob("train.csv"):
            print(f"Using discovered Kaggle dataset path: {found}")
            return found

    return data_path if data_path.is_absolute() else (PROJECT_ROOT / data_path)


def _resolve_output_dir(configured_dir: str) -> Path:
    output_dir = Path(configured_dir)
    if output_dir.is_absolute():
        return output_dir
    if str(PROJECT_ROOT).startswith("/kaggle/input"):
        return Path("/kaggle/working") / output_dir
    return PROJECT_ROOT / output_dir


def _resolve_mlflow_tracking_uri(tracking_uri: str) -> str:
    if "://" in tracking_uri and not tracking_uri.startswith("file:"):
        return tracking_uri

    if tracking_uri.startswith("file:"):
        local_path = Path(tracking_uri[5:])
    else:
        local_path = Path(tracking_uri)

    if not local_path.is_absolute():
        local_path = PROJECT_ROOT / local_path

    local_path = local_path.resolve()
    local_path.mkdir(parents=True, exist_ok=True)
    return f"file:{local_path.as_posix()}"


def _setup_mlflow(config: Dict[str, Any]) -> Dict[str, Any]:
    mlflow_cfg = config.get("mlflow", {}) or {}
    if not mlflow_cfg.get("enabled", False):
        return {"enabled": False}

    try:
        import mlflow   
    except Exception as exc:
        print(f"MLflow is enabled in config but could not installed {exc}")
        return {"enabled": False}

    tracking_uri = _resolve_mlflow_tracking_uri(str(mlflow_cfg.get("tracking_uri", "mlruns")))
    experiment_name = str(mlflow_cfg.get("experiment_name", "mlops_a2_gan"))
    run_name = mlflow_cfg.get("run_name") or None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return {
        "enabled": True,
        "mlflow": mlflow,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "run_name": run_name,
    }


_bootstrap_import_path()

try:
    from src.config import load_config, resolve_device
    from src.data import load_and_preprocess_data
    from src.models import Discriminator, Generator
    from src.training import GANTrainer
    from src.utils import (
        evaluate_discriminator_confidence,
        generate_samples,
        plot_generated_digits,
        plot_loss_curves,
        plot_real_vs_fake,
        print_evaluation_report,
    )
except ModuleNotFoundError as exc:
    if exc.name == "src" or (exc.name and exc.name.startswith("src.")):
        raise RuntimeError(
            "Could not import 'src'. In Kaggle script kernels, attach project files "
            "as a dataset source so runtime includes src/ and config.yaml."
        ) from exc
    raise

def main() -> None:
    """Run the code workflow based on a YAML config file."""
    parser = argparse.ArgumentParser(description="Train a GAN on MNIST digits")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    config = load_config(str(config_path))
    data_config = config["data"]
    training_config = config["training"]
    runtime_config = config["runtime"]
    output_config = config["output"]

    output_dir = _resolve_output_dir(output_config["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow_state = _setup_mlflow(config)
    mlflow = mlflow_state.get("mlflow")

    print(f"Project root: {PROJECT_ROOT}")
    device = resolve_device(runtime_config["device"])
    print(f"Using device: {device}")
    if mlflow_state.get("enabled"):
        print(
            "MLflow enabled: "
            f"experiment={mlflow_state['experiment_name']}, "
            f"tracking_uri={mlflow_state['tracking_uri']}"
        )

    print("Loading and preprocessing data...")
    data_path = _resolve_data_path(data_config["path"])
    print(f"Using data path: {data_path}")
    data_loader = load_and_preprocess_data(
        str(data_path),
        batch_size=training_config["batch_size"],
    )

    print("Initializing Generator and Discriminator...")
    generator = Generator(latent_dim=training_config["latent_dim"])
    discriminator = Discriminator()
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        lr=training_config["learning_rate"],
        device=device,
    )

    def _on_epoch_end(epoch_idx: int, g_loss: float, d_loss: float) -> None:
        if mlflow is None:
            return
        mlflow.log_metric("generator_loss", float(g_loss), step=epoch_idx)
        mlflow.log_metric("discriminator_loss", float(d_loss), step=epoch_idx)

    mlflow_run_context = (
        mlflow.start_run(run_name=mlflow_state.get("run_name"))
        if mlflow is not None
        else nullcontext()
    )
    with mlflow_run_context:
        if mlflow is not None:
            mlflow.log_params(
                {
                    "data_path": str(data_path),
                    "device": device,
                    "epochs": int(training_config["epochs"]),
                    "batch_size": int(training_config["batch_size"]),
                    "learning_rate": float(training_config["learning_rate"]),
                    "latent_dim": int(training_config["latent_dim"]),
                    "num_samples": int(output_config["num_samples"]),
                }
            )
            mlflow.log_artifact(str(config_path), artifact_path="config")

        print(f"Training for {training_config['epochs']} epochs...")
        trainer.train(
            data_loader,
            epochs=training_config["epochs"],
            verbose=True,
            metric_callback=_on_epoch_end if mlflow is not None else None,
        )

        print("\nGenerating visualizations...")
        real_images = next(iter(data_loader))[0]
        samples = generate_samples(
            generator,
            n_samples=output_config["num_samples"],
            device=device,
        )

        generated_path = output_dir / "generated_digits.png"
        losses_path = output_dir / "loss_curves.png"
        real_vs_fake_path = output_dir / "real_vs_fake.png"
        training_log_path = output_dir / "training_log.txt"

        if trainer.training_log:
            training_log_path.write_text("\n".join(trainer.training_log) + "\n", encoding="utf-8")

        plot_generated_digits(samples, save_path=str(generated_path))
        plot_loss_curves(
            trainer.g_loss_history,
            trainer.d_loss_history,
            save_path=str(losses_path),
        )
        plot_real_vs_fake(
            real_images,
            generator,
            device=device,
            save_path=str(real_vs_fake_path),
        )

        real_score, fake_score = evaluate_discriminator_confidence(
            discriminator,
            generator,
            real_images,
            device=device,
        )
        print_evaluation_report(real_score, fake_score)

        if mlflow is not None:
            mlflow.log_metric("real_score", float(real_score))
            mlflow.log_metric("fake_score", float(fake_score))
            for artifact_path in (generated_path, losses_path, real_vs_fake_path):
                mlflow.log_artifact(str(artifact_path), artifact_path="plots")
            if training_log_path.exists():
                mlflow.log_artifact(str(training_log_path), artifact_path="logs")

    print(f"\nDone! Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
