import requests, json

BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)
ORDERS_URL = "{}/v2/orders".format(BASE_URL)
HEADERS = {'APCA-API-KEY-ID': 'PKDCNL5G6CQ3QW1GUWNO', 'APCA-API-SECRET-KEY': 'yRX52EMDOpxJAEWKiYnzFcb7ByMhYqzT9jmnp26e'}

def get_account():
    r = requests.get(ACCOUNT_URL, headers = HEADERS)

    return json.loads(r.content)

def create_order(symbol, qty, side, type, time_in_force, limit_price):
    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "limit",
        "time_in_force": time_in_force,
        "limit_price": limit_price
    }

    r = requests.post(ORDERS_URL, json = data, headers = HEADERS)

    return json.loads(r.content)

def main():
    # Algorithm for creating orders
    # Order documentation: https://alpaca.markets/docs/api-documentation/how-to/orders/

    # Psuedo code, want to place a limit order for the day.
        # If we have money and the limit condition is met, BUY
        # If we only have stocks and the limit condition is met, SELL


if __name__ == "__main__":
    main()
