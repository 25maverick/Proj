from vetiver import VetiverModel
from dotenv import load_dotenv, find_dotenv
import vetiver
import pins

load_dotenv(find_dotenv())

b = pins.board_folder('/users/weissyuan/M378', allow_pickle_read=True)
v = VetiverModel.from_pin(b, 'penguin_model', version = '20240423T162820Z-84604')

vetiver_api = vetiver.VetiverAPI(v)
api = vetiver_api.app
