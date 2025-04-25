from telegram import ReplyKeyboardMarkup


class KeyboardService:
    @staticmethod
    def get_main_menu():
        return ReplyKeyboardMarkup([
            ['Каталог', 'Акции'],
            ['Корзина', 'Помощь']
        ], resize_keyboard=True)

    @staticmethod
    def get_furniture_menu():
        return ReplyKeyboardMarkup([
            ['Диваны', 'Кресла'],
            ['Шкафы', 'Столы'],
            ['Назад']
        ], resize_keyboard=True)