from flask import Flask, request
import chdb
import os

# chdb API server example with GET/POST support, compatible with play app
# for a full server example see https://github.com/metrico/chdb-server

app = Flask(__name__, static_folder="", static_url_path="")

@app.route('/', methods=["GET"])
def clickhouse():
    query = request.args.get('query', default="", type=str)
    format = request.args.get('default_format', default="JSONCompact", type=str)
    if not query:
        return "Query not found", 400

    res = chdb.query(query, format)
    return res.get_memview().tobytes()

@app.route('/', methods=["POST"])
def play():
    query = request.data
    format = request.args.get('default_format', default="JSONCompact", type=str)
    if not query:
        return "Query not found", 400

    res = chdb.query(query, format)
    return res.get_memview().tobytes()

@app.errorhandler(404)
def handle_404(e):
    return "Not found", 404

host = os.getenv('HOST', '0.0.0.0')
port = os.getenv('PORT', 8123)
app.run(host=host, port=port)
