from abc import ABCMeta, abstractmethod


class AbstractMessagingInterface(metaclass=ABCMeta):

    @abstractmethod
    def send_data(self, type_of_data, data):
        pass

    @abstractmethod
    def authenticate(self, auth_body):
        pass

    @abstractmethod
    def get_data(self, data_type, data_request_body):
        pass
