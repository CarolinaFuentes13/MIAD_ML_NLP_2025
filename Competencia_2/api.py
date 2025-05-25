#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_deployment import predict_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Géneros de pelicula',
    description='Esta API devuelve los géneros a los que puede pertenecer una pelicula basada en su sinopsis y título')

ns = api.namespace('predict', 
     description='Género')
   
parser = api.parser()


parser.add_argument('title', type=str, required=True, location = 'args')
parser.add_argument('plot', type=str, required=True, location = 'args')



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
            title=args['title'],
            plot=args['plot'],

        )
        
        return {"result": result}, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)