import run
import jinja2
import json
from flask import Flask, render_template
from koneksi import ConDB

app = Flask(__name__)


@app.route('/')
def index():
    return "coba"


@app.route('/optimization/<strs>')
def dashboard(strs):
    template = jinja2.Template('{{test}}')
    t = str(strs)
    tourid = []
    t_ = json.loads(t)
    for t in t_['idWisata']:
        tourid.append(t['id'])
    idhotel = int(t_['idhotel'][0])
    dwaktu = float(t_['degree'][0])
    dtarif = float(t_['degree'][2])
    drating = float(t_['degree'][1])
    travel_days = int(t_['travel_days'])
    algorithm = str(t_['algo_url'])
    
    itinerary, fitness = run.main(tourid, idhotel, dwaktu, drating, dtarif, travel_days, algorithm)
    # print('itenenary')
    # print('------------------------')
    # print(itinerary)
    # print('------------------------')
    dictionary = {'results': itinerary}

    # print(dictionary)

#    json dumps ngerubah dictionary jadi json ntar hasil algoritma dictionari terus jadiin json
    return render_template(template,test=json.dumps(dictionary))


if __name__ == "__main__":
    app.run()
