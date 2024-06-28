import argparse
import os
import requests
import re
import subprocess
import shutil

model_name = "https://civitai.com/api/download/models/147497"

# Add in config option to change root folder

BASE_PATH = os.path.expanduser("~/civitai")
MODEL_PATH = os.path.expanduser("~/civitai/models")
TEMP_PATH = os.path.expanduser("~/civitai/temp")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)


if __name__ == "__main__":
    # Retrieve args
    argp = argparse.ArgumentParser(description="Miner Configs")
    argp.add_argument("--civitai_link", type=str, default=None)
    argp.add_argument("--is_sdxl", action="store_true")
    pa = argp.parse_args()

    assert (
        pa.civitai_link
    ), "You must pass --civitai_link with the link to the desired model."

    model_name = pa.civitai_link
    # Download civitai model to temp
    file_name = model_name.rsplit("/", 1)[-1]
    file_path = f"{TEMP_PATH}/{file_name}"

    print(f"Downloading {model_name} to {file_path}.")
    response = requests.get(model_name)

    original_name = (
        re.findall("filename=(.+)", response.headers["content-disposition"])[0]
        .replace('"', "")
        .replace("'", "")
    )

    with open(file_path, "wb") as f:
        f.write(response.content)

    # Get which version of diffusers is used
    stoutdata = subprocess.getoutput("pip freeze | grep diffusers==").split("\n")
    version = None
    for line in stoutdata:
        if line.startswith("diffusers=="):
            version = line.split("==")[-1]

    # Convert to numerical version for comparison
    versions = [int(x) for x in version.split(".")]
    numerical_version = 0
    for i, v in enumerate(versions):
        numerical_version += int(v * (10000 / (10 ** (i + 1))))
    assert (
        numerical_version >= 2140
    ), "Please install diffusers version 0.21.4 or higher: pip install diffusers>=0.21.4"

    # Download the conversion script for the version passed:
    script_url = f"https://raw.githubusercontent.com/huggingface/diffusers/v{version}/scripts/convert_original_stable_diffusion_to_diffusers.py"
    script_path = f"{BASE_PATH}/conversion_{version}.py"
    print(f"Downloading conversion script {script_url} to {script_path}.")
    download_response = requests.get(script_url)

    assert (
        download_response.status_code == 200
    ), f"An error occurred downloading the conversion script: Status code: {download_response.status_code}"

    with open(script_path, "wb") as f:
        f.write(download_response.content)

    # Apply patch to fix bug
    print("Applying patch for 0.21.4.")
    with open(script_path, "r") as file_in:
        content = file_in.read().replace("config_files=args.config_files,", "")
        with open(script_path, "w") as file_out:
            file_out.write(content)

    # Set the output path
    output_name = original_name.replace(".safetensors", "")
    output_path = f"{MODEL_PATH}/{output_name}/"

    # Execute the conversion script
    command = f"python {script_path} --checkpoint_path {file_path} --dump_path {output_path} --from_safetensors"

    if pa.is_sdxl:
        print("Using StableDiffusionXLPipeline.")
        command += " --pipeline_class_name StableDiffusionXLPipeline"

    # Run
    print(f"Converting with command: {command}.")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

    # Append .failed to the end of the folder name
    if process.returncode != 0:
        print("Appended the suffix 'failed' to the output.")
        shutil.move(output_path, output_path.rstrip("/") + ".failed")

    assert (
        process.returncode == 0
    ), f"An error occurred trying to convert the model: Return code: {process.returncode}"

    print("The process has finished successfully.")
