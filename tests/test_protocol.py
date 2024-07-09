import base64
import os

import torch
from loguru import logger

from neurons.protocol import ImageGeneration

os.environ["USE_TORCH"] = "1"

incoming_synapse_new_base64 = {
    "name": "ImageGeneration",
    "timeout": 20.0,
    "total_size": 4614,
    "header_size": 0,
    "dendrite": {
        "status_code": None,
        "status_message": None,
        "process_time": None,
        "ip": "1.2.3.4",
        "port": None,
        "version": 7002000,
        "nonce": 1720514029378242357,
        "uuid": "edf1f5b0-3dcd-11ef-ada1-311ff3520135",
        "hotkey": "fake_hotkey",
        "signature": "0x1234",
    },
    "axon": {
        "status_code": None,
        "status_message": None,
        "process_time": None,
        "ip": "1.2.3.4",
        "port": 8101,
        "version": None,
        "nonce": None,
        "uuid": None,
        "hotkey": "fake_hotkey",
        "signature": None,
    },
    "computed_body_hash": "",
    "required_hash_fields": [],
    "prompt": "In the serene countryside, an old unicorn magically restores a rusty car, transforming it back to its elegant, vintage glory.",
    "negative_prompt": None,
    "prompt_image": None,
    "images": [
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z/C/HgAGgwJ/lK3Q6wAAAABJRU5ErkJggg=="
    ],
    "num_images_per_prompt": 1,
    "height": 1024,
    "width": 1024,
    "generation_type": "text_to_image",
    "guidance_scale": 7.5,
    "seed": -1,
    "steps": 50,
    "model_type": "CUSTOM",
}

incoming_synapse_old = {
    "name": "ImageGeneration",
    "timeout": 20.0,
    "total_size": 4639,
    "header_size": 0,
    "dendrite": {
        "status_code": None,
        "status_message": None,
        "process_time": None,
        "ip": "1.2.3.4",
        "port": None,
        "version": 7002000,
        "nonce": 1720518607255466062,
        "uuid": "0f10708c-3dd8-11ef-ada1-311ff3520135",
        "hotkey": "fake_hotkey",
        "signature": "0x1234",
    },
    "axon": {
        "status_code": None,
        "status_message": None,
        "process_time": None,
        "ip": "1.2.3.4",
        "port": 8101,
        "version": None,
        "nonce": None,
        "uuid": None,
        "hotkey": "fake_hotkey",
        "signature": None,
    },
    "computed_body_hash": "",
    "required_hash_fields": [],
    "prompt": "A playful griffin teasingly flies around a bustling stadium, holding a giant pair of golden scissors in its talons, ready to cut the inaugural ribbon.",
    "negative_prompt": None,
    "prompt_image": None,
    "images": [
        {
            "buffer": "hcQCbmTDxAR0eXBlo3x1McQEa2luZMQAxAVzaGFwZZMDAQHEBGRhdGHEAwAAAA==",
            "dtype": "torch.uint8",
            "shape": [3, 1, 1],
        }
    ],
    "num_images_per_prompt": 1,
    "height": 1024,
    "width": 1024,
    "generation_type": "text_to_image",
    "guidance_scale": 7.5,
    "seed": -1,
    "steps": 50,
    "model_type": "CUSTOM",
}


def test_deserialize_old_miner_synapse():
    synapse = ImageGeneration(**incoming_synapse_old)
    assert type(synapse.images[0]) == str
    # Should be able to decode
    base64.b64decode(synapse.images[0])


def test_deserialize_new_miner_synapse():
    synapse = ImageGeneration(**incoming_synapse_new_base64)
    assert type(synapse.images[0]) == str
    # Should be able to decode
    base64.b64decode(synapse.images[0])
