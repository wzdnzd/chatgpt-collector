# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-08-13

import utils
from provider.librechat import LibreChat
from provider.oneapi import OneAPI
from provider.openwebui import OpenWebUI

SUPPORTED_PROVIDERS = {OpenWebUI._name(): OpenWebUI, OneAPI._name(): OneAPI, LibreChat._name(): LibreChat}


def get_provider(service: str, domain: str) -> OpenWebUI:
    service = utils.trim(service).lower()
    if service not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported service provider: {service}")

    return SUPPORTED_PROVIDERS[service](domain=domain)
