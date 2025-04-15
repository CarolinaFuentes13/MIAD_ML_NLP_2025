#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_deployment import predict_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Popularity Prediction API',
    description='Song popularity API')

ns = api.namespace('predict', 
     description='Popularity')
   
parser = api.parser()


parser.add_argument('artists', type=str, required=True, location = 'args')
parser.add_argument('album_name', type=str, required=True, location = 'args')
parser.add_argument('track_name', type=str, required=True, location = 'args')
parser.add_argument('duration_ms', type=int, required=True, location = 'args')
parser.add_argument('explicit', type=bool, required=True, location = 'args')
parser.add_argument('danceability', type=float, required=True, location = 'args')
parser.add_argument('energy', type=float, required=True, location = 'args')
parser.add_argument('key', type=int, required=True, location = 'args')
parser.add_argument('loudness', type=float, required=True, location = 'args')
parser.add_argument('mode', type=int, required=True, location = 'args')
parser.add_argument('speechiness', type=float, required=True, location = 'args')
parser.add_argument('acousticness', type=float, required=True, location = 'args')
parser.add_argument('instrumentalness', type=float, required=True, location = 'args')
parser.add_argument('liveness', type=float, required=True, location = 'args')
parser.add_argument('valence', type=float, required=True, location = 'args')
parser.add_argument('tempo', type=float, required=True, location = 'args')
parser.add_argument('time_signature', type=int, required=True, location = 'args')
parser.add_argument('track_genre', type=str, required=True, location = 'args')


resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        result = predict_proba(
            artists=args['artists'],
            album_name=args['album_name'],
            track_name=args['track_name'],
            duration_ms=args['duration_ms'],
            explicit=args['explicit'],
            danceability=args['danceability'],
            energy=args['energy'],
            key=args['key'],
            loudness=args['loudness'],
            mode=args['mode'],
            speechiness=args['speechiness'],
            acousticness=args['acousticness'],
            instrumentalness=args['instrumentalness'],
            liveness=args['liveness'],
            valence=args['valence'],
            tempo=args['tempo'],
            time_signature=args['time_signature'],
            track_genre=args['track_genre']
        )
        
        return {"result": result}, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)