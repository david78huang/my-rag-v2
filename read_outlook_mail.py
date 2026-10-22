#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read recent Outlook emails through Microsoft Graph.

Usage:
  python read_outlook_mail.py --client_id <AZURE_APP_CLIENT_ID>

Or:
  set OUTLOOK_CLIENT_ID=<AZURE_APP_CLIENT_ID>
  python read_outlook_mail.py
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

import requests

GRAPH_ROOT = "https://graph.microsoft.com/v1.0"


def post_form(url: str, data: Dict[str, str]) -> Dict[str, Any]:
    response = requests.post(url, data=data, timeout=30)
    try:
        payload = response.json()
    except ValueError:
        response.raise_for_status()
        raise
    if not response.ok:
        message = payload.get("error_description") or payload.get("error") or response.text
        raise RuntimeError(message)
    return payload


def get_json(url: str, token: str) -> Dict[str, Any]:
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    try:
        payload = response.json()
    except ValueError:
        response.raise_for_status()
        raise
    if not response.ok:
        message = payload.get("error", {}).get("message") or response.text
        raise RuntimeError(message)
    return payload


def get_device_code(client_id: str, tenant: str, scopes: str) -> Dict[str, Any]:
    return post_form(
        f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode",
        {
            "client_id": client_id,
            "scope": scopes,
        },
    )


def wait_for_token(client_id: str, tenant: str, device_code: str, interval: int) -> str:
    token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    while True:
        payload = requests.post(
            token_url,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": client_id,
                "device_code": device_code,
            },
            timeout=30,
        ).json()

        if "access_token" in payload:
            return payload["access_token"]

        error = payload.get("error")
        if error == "authorization_pending":
            time.sleep(interval)
            continue
        if error == "slow_down":
            interval += 5
            time.sleep(interval)
            continue

        message = payload.get("error_description") or str(payload)
        raise RuntimeError(message)


def acquire_token(client_id: str, tenant: str) -> str:
    scopes = "User.Read Mail.Read offline_access"
    device = get_device_code(client_id, tenant, scopes)
    print(device["message"])
    return wait_for_token(
        client_id=client_id,
        tenant=tenant,
        device_code=device["device_code"],
        interval=int(device.get("interval", 5)),
    )


def build_messages_url(folder: str, top: int, unread_only: bool) -> str:
    folder_path = "me/messages" if folder.lower() == "inbox" else f"me/mailFolders/{folder}/messages"
    fields = ",".join(
        [
            "id",
            "subject",
            "from",
            "receivedDateTime",
            "isRead",
            "bodyPreview",
            "webLink",
        ]
    )
    query = f"$top={top}&$orderby=receivedDateTime desc&$select={fields}"
    if unread_only:
        query += "&$filter=isRead eq false"
    return f"{GRAPH_ROOT}/{folder_path}?{query}"


def sender_name(message: Dict[str, Any]) -> str:
    email = message.get("from", {}).get("emailAddress", {})
    name = email.get("name") or ""
    address = email.get("address") or ""
    if name and address:
        return f"{name} <{address}>"
    return name or address or "(unknown sender)"


def print_messages(messages: Dict[str, Any]) -> None:
    for index, message in enumerate(messages.get("value", []), start=1):
        print("=" * 80)
        print(f"{index}. {message.get('subject') or '(no subject)'}")
        print(f"From: {sender_name(message)}")
        print(f"Received: {message.get('receivedDateTime')}")
        print(f"Read: {message.get('isRead')}")
        print(f"Link: {message.get('webLink')}")
        preview = (message.get("bodyPreview") or "").replace("\r", " ").replace("\n", " ")
        print(f"Preview: {preview[:500]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read recent Outlook emails through Microsoft Graph.")
    parser.add_argument("--client_id", default=os.getenv("OUTLOOK_CLIENT_ID"))
    parser.add_argument("--tenant", default=os.getenv("OUTLOOK_TENANT", "common"))
    parser.add_argument("--access_token", default=os.getenv("OUTLOOK_ACCESS_TOKEN"))
    parser.add_argument("--folder", default="inbox")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--unread_only", action="store_true")
    args = parser.parse_args()

    token: Optional[str] = args.access_token
    if not token:
        if not args.client_id:
            raise SystemExit(
                "Missing --client_id or OUTLOOK_CLIENT_ID. Register an Azure app and use its Application "
                "(client) ID. The app needs delegated Mail.Read permission and public-client/device-code login."
            )
        token = acquire_token(args.client_id, args.tenant)

    messages = get_json(build_messages_url(args.folder, args.top, args.unread_only), token)
    print_messages(messages)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
