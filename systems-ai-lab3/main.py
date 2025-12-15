"""
Лабораторная работа №3: Нечёткие модели
Часть 1: Управление умным домом
Часть 2: Моделирование нелинейных функций (с линейными функциями TSK)
"""

import sys
import numpy as np
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QTextEdit,
                             QGroupBox, QRadioButton, QComboBox, QMessageBox, QTabWidget,
                             QFrame, QSizePolicy, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class FuzzySet:
    def __init__(self, name, universe, membership_func):
        self.name = name
        self.universe = universe
        self.membership_func = membership_func
        self.membership_values = membership_func(universe)
    
    def get_membership(self, x):
        return self.membership_func(x)

def trimf(x, params):
    a, b, c = params
    return np.maximum(0, np.minimum((x-a)/(b-a+1e-8), (c-x)/(c-b+1e-8)))

def trapmf(x, params):
    a, b, c, d = params
    return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a+1e-8), 1), (d-x)/(d-c+1e-8)))

class FuzzyVariable:
    def __init__(self, name, universe):
        self.name = name
        self.universe = universe
        self.terms = {}
    
    def add_term(self, term_name, membership_func):
        self.terms[term_name] = FuzzySet(term_name, self.universe, membership_func)
    
    def get_term(self, term_name):
        return self.terms[term_name]

class FuzzyRule:
    def __init__(self, rule_id, antecedents, consequent):
        self.rule_id = rule_id
        self.antecedents = antecedents
        self.consequent = consequent
    
    def __repr__(self):
        ant_str = " AND ".join([f"{v} IS {t}" for v, t in self.antecedents.items()])
        cons_str = f"{self.consequent[0]} IS {self.consequent[1]}"
        return f"Rule {self.rule_id}: IF {ant_str} THEN {cons_str}"

class FuzzyOperators:
    @staticmethod
    def min_tnorm(a, b): return np.minimum(a, b)
    @staticmethod
    def prod_tnorm(a, b): return a * b
    @staticmethod
    def max_snorm(a, b): return np.maximum(a, b)
    @staticmethod
    def probsum_snorm(a, b): return a + b - a * b

class Implication:
    @staticmethod
    def mamdani(degree, mf): return np.minimum(degree, mf)
    @staticmethod
    def larsen(degree, mf): return degree * mf
    @staticmethod
    def lukasiewicz(degree, mf): return np.minimum(1, 1 - degree + mf)

class FuzzyInferenceSystem:
    def __init__(self, name="FIS"):
        self.name = name
        self.input_variables = {}
        self.output_variables = {}
        self.rules = []
        self.implication_method = Implication.mamdani
        self.aggregation_method = FuzzyOperators.max_snorm
        self.tnorm_method = FuzzyOperators.min_tnorm
    
    def add_input_variable(self, var):
        self.input_variables[var.name] = var
    
    def add_output_variable(self, var):
        self.output_variables[var.name] = var
    
    def add_rule(self, rule):
        self.rules.append(rule)
    
    def set_implication(self, method):
        methods = {'mamdani': Implication.mamdani, 'larsen': Implication.larsen, 'lukasiewicz': Implication.lukasiewicz}
        self.implication_method = methods.get(method.lower(), Implication.mamdani)
    
    def set_aggregation(self, method):
        methods = {'max': FuzzyOperators.max_snorm, 'probsum': FuzzyOperators.probsum_snorm}
        self.aggregation_method = methods.get(method.lower(), FuzzyOperators.max_snorm)
    
    def set_tnorm(self, method):
        methods = {'min': FuzzyOperators.min_tnorm, 'prod': FuzzyOperators.prod_tnorm}
        self.tnorm_method = methods.get(method.lower(), FuzzyOperators.min_tnorm)
    
    def fuzzify(self, inputs):
        fuzzified = {}
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                fuzzified[var_name] = {
                    term_name: fuzzy_set.get_membership(value)
                    for term_name, fuzzy_set in self.input_variables[var_name].terms.items()
                }
        return fuzzified
    
    def evaluate_rules(self, fuzzified_inputs):
        activated_rules = []
        for rule in self.rules:
            degrees = []
            for var, term in rule.antecedents.items():
                if var in fuzzified_inputs and term in fuzzified_inputs[var]:
                    degrees.append(fuzzified_inputs[var][term])
            
            if degrees:
                combined = degrees[0]
                for d in degrees[1:]:
                    combined = self.tnorm_method(combined, d)
                
                out_var, out_term = rule.consequent
                if out_var in self.output_variables:
                    mf = self.output_variables[out_var].get_term(out_term).membership_values
                    implied = self.implication_method(combined, mf)
                    activated_rules.append((rule, combined, implied))
        
        return activated_rules
    
    def aggregate(self, activated_rules, output_var_name):
        if not activated_rules:
            return np.zeros_like(self.output_variables[output_var_name].universe)
        
        aggregated = activated_rules[0][2]
        for _, _, implied_mf in activated_rules[1:]:
            aggregated = self.aggregation_method(aggregated, implied_mf)
        
        return aggregated
    
    def defuzzify_centroid(self, aggregated_mf, output_var_name):
        universe = self.output_variables[output_var_name].universe
        numerator = np.sum(universe * aggregated_mf)
        denominator = np.sum(aggregated_mf)
        return np.mean(universe) if denominator == 0 else numerator / denominator
    
    def inference(self, inputs, defuzz_method='centroid'):
        fuzzified = self.fuzzify(inputs)
        activated_rules = self.evaluate_rules(fuzzified)
        output_var_name = list(self.output_variables.keys())[0]
        aggregated = self.aggregate(activated_rules, output_var_name)
        crisp_output = self.defuzzify_centroid(aggregated, output_var_name)
        return crisp_output, aggregated, activated_rules
    
    def inference_tsk(self, inputs):
        fuzzified = self.fuzzify(inputs)
        weighted_sum = 0
        weight_sum = 0
        
        for rule in self.rules:
            degrees = []
            for var, term in rule.antecedents.items():
                if var in fuzzified and term in fuzzified[var]:
                    degrees.append(fuzzified[var][term])
            
            if degrees:
                activation = degrees[0]
                for d in degrees[1:]:
                    activation = self.tnorm_method(activation, d)
                
                out_var, out_term = rule.consequent
                if out_var in self.output_variables:
                    term_obj = self.output_variables[out_var].get_term(out_term)
                    universe = term_obj.universe
                    mf = term_obj.membership_values
                    center = np.sum(universe * mf) / (np.sum(mf) + 1e-10)
                    
                    weighted_sum += activation * center
                    weight_sum += activation
        
        return weighted_sum / (weight_sum + 1e-10) if weight_sum > 0 else 0
    
    def save_to_file(self, filename):
        data = {'name': self.name, 'input_variables': {}, 'output_variables': {}, 'rules': []}
        
        for var_name, var in self.input_variables.items():
            data['input_variables'][var_name] = {
                'universe': [float(var.universe[0]), float(var.universe[-1]), len(var.universe)],
                'terms': list(var.terms.keys())
            }
        
        for var_name, var in self.output_variables.items():
            data['output_variables'][var_name] = {
                'universe': [float(var.universe[0]), float(var.universe[-1]), len(var.universe)],
                'terms': list(var.terms.keys())
            }
        
        for rule in self.rules:
            data['rules'].append({
                'id': rule.rule_id,
                'antecedents': rule.antecedents,
                'consequent': list(rule.consequent)
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def create_lighting_control_system():
    illumination = FuzzyVariable('освещённость', np.linspace(0, 100, 200))
    illumination.add_term('темно', lambda x: trapmf(x, [0, 0, 10, 25]))
    illumination.add_term('сумерки', lambda x: trimf(x, [15, 30, 45]))
    illumination.add_term('умеренно', lambda x: trimf(x, [35, 50, 65]))
    illumination.add_term('светло', lambda x: trapmf(x, [55, 70, 100, 100]))
    
    brightness = FuzzyVariable('яркость', np.linspace(0, 100, 200))
    brightness.add_term('выключен', lambda x: trapmf(x, [0, 0, 5, 15]))
    brightness.add_term('минимальная', lambda x: trimf(x, [10, 25, 40]))
    brightness.add_term('средняя', lambda x: trimf(x, [30, 50, 70]))
    brightness.add_term('высокая', lambda x: trimf(x, [60, 75, 90]))
    brightness.add_term('максимальная', lambda x: trapmf(x, [80, 90, 100, 100]))
    
    fis = FuzzyInferenceSystem('Управление_освещением_умного_дома')
    fis.add_input_variable(illumination)
    fis.add_output_variable(brightness)
    
    rules = [
        FuzzyRule(1, {'освещённость': 'темно'}, ('яркость', 'максимальная')),
        FuzzyRule(2, {'освещённость': 'темно'}, ('яркость', 'высокая')),
        FuzzyRule(3, {'освещённость': 'сумерки'}, ('яркость', 'высокая')),
        FuzzyRule(4, {'освещённость': 'сумерки'}, ('яркость', 'средняя')),
        FuzzyRule(5, {'освещённость': 'умеренно'}, ('яркость', 'средняя')),
        FuzzyRule(6, {'освещённость': 'умеренно'}, ('яркость', 'минимальная')),
        FuzzyRule(7, {'освещённость': 'светло'}, ('яркость', 'минимальная')),
        FuzzyRule(8, {'освещённость': 'светло'}, ('яркость', 'выключен')),
    ]
    
    for rule in rules:
        fis.add_rule(rule)
    
    return fis

def create_climate_control_system_3():
    temperature = FuzzyVariable('температура', np.linspace(10, 35, 200))
    temperature.add_term('холодно', lambda x: trapmf(x, [10, 10, 14, 18]))
    temperature.add_term('прохладно', lambda x: trimf(x, [16, 19, 22]))
    temperature.add_term('комфортно', lambda x: trimf(x, [20, 23, 26]))
    temperature.add_term('тепло', lambda x: trimf(x, [24, 27, 30]))
    temperature.add_term('жарко', lambda x: trapmf(x, [28, 31, 35, 35]))
    
    humidity = FuzzyVariable('влажность', np.linspace(20, 80, 200))
    humidity.add_term('сухо', lambda x: trapmf(x, [20, 20, 30, 42]))
    humidity.add_term('нормально', lambda x: trimf(x, [38, 50, 62]))
    humidity.add_term('влажно', lambda x: trapmf(x, [58, 68, 80, 80]))
    
    time_of_day = FuzzyVariable('время_суток', np.linspace(0, 23, 48))
    time_of_day.add_term('ночь', lambda x: trapmf(x, [0, 0, 3, 6]))
    time_of_day.add_term('утро', lambda x: trimf(x, [5, 8, 11]))
    time_of_day.add_term('день', lambda x: trimf(x, [10, 14, 18]))
    time_of_day.add_term('вечер', lambda x: trimf(x, [17, 20, 23]))
    
    power = FuzzyVariable('мощность', np.linspace(-100, 100, 300))
    power.add_term('сильное_охлаждение', lambda x: trapmf(x, [-100, -100, -80, -60]))
    power.add_term('охлаждение', lambda x: trimf(x, [-70, -50, -30]))
    power.add_term('выключено', lambda x: trimf(x, [-15, 0, 15]))
    power.add_term('обогрев', lambda x: trimf(x, [30, 50, 70]))
    power.add_term('сильный_обогрев', lambda x: trapmf(x, [60, 80, 100, 100]))
    
    fis = FuzzyInferenceSystem('Климат_контроль_умного_дома')
    fis.add_input_variable(temperature)
    fis.add_input_variable(humidity)
    fis.add_input_variable(time_of_day)
    fis.add_output_variable(power)
    
    rules = [
        FuzzyRule(1, {'температура': 'холодно', 'влажность': 'сухо', 'время_суток': 'утро'}, ('мощность', 'сильный_обогрев')),
        FuzzyRule(2, {'температура': 'холодно', 'влажность': 'нормально', 'время_суток': 'утро'}, ('мощность', 'сильный_обогрев')),
        FuzzyRule(3, {'температура': 'холодно', 'влажность': 'влажно', 'время_суток': 'утро'}, ('мощность', 'обогрев')),
        FuzzyRule(4, {'температура': 'прохладно', 'влажность': 'сухо', 'время_суток': 'утро'}, ('мощность', 'обогрев')),
        FuzzyRule(5, {'температура': 'прохладно', 'влажность': 'нормально', 'время_суток': 'утро'}, ('мощность', 'обогрев')),
        FuzzyRule(6, {'температура': 'комфортно', 'влажность': 'нормально', 'время_суток': 'утро'}, ('мощность', 'выключено')),
        FuzzyRule(7, {'температура': 'тепло', 'влажность': 'нормально', 'время_суток': 'утро'}, ('мощность', 'выключено')),
        FuzzyRule(8, {'температура': 'холодно', 'влажность': 'сухо', 'время_суток': 'день'}, ('мощность', 'обогрев')),
        FuzzyRule(9, {'температура': 'холодно', 'влажность': 'нормально', 'время_суток': 'день'}, ('мощность', 'обогрев')),
        FuzzyRule(10, {'температура': 'прохладно', 'влажность': 'нормально', 'время_суток': 'день'}, ('мощность', 'обогрев')),
        FuzzyRule(11, {'температура': 'комфортно', 'влажность': 'сухо', 'время_суток': 'день'}, ('мощность', 'выключено')),
        FuzzyRule(12, {'температура': 'комфортно', 'влажность': 'нормально', 'время_суток': 'день'}, ('мощность', 'выключено')),
        FuzzyRule(13, {'температура': 'комфортно', 'влажность': 'влажно', 'время_суток': 'день'}, ('мощность', 'выключено')),
        FuzzyRule(14, {'температура': 'тепло', 'влажность': 'сухо', 'время_суток': 'день'}, ('мощность', 'выключено')),
        FuzzyRule(15, {'температура': 'тепло', 'влажность': 'нормально', 'время_суток': 'день'}, ('мощность', 'охлаждение')),
        FuzzyRule(16, {'температура': 'жарко', 'влажность': 'сухо', 'время_суток': 'день'}, ('мощность', 'охлаждение')),
        FuzzyRule(17, {'температура': 'жарко', 'влажность': 'нормально', 'время_суток': 'день'}, ('мощность', 'сильное_охлаждение')),
        FuzzyRule(18, {'температура': 'жарко', 'влажность': 'влажно', 'время_суток': 'день'}, ('мощность', 'сильное_охлаждение')),
        FuzzyRule(19, {'температура': 'холодно', 'влажность': 'сухо', 'время_суток': 'вечер'}, ('мощность', 'обогрев')),
        FuzzyRule(20, {'температура': 'холодно', 'влажность': 'нормально', 'время_суток': 'вечер'}, ('мощность', 'обогрев')),
        FuzzyRule(21, {'температура': 'холодно', 'влажность': 'влажно', 'время_суток': 'вечер'}, ('мощность', 'обогрев')),
        FuzzyRule(22, {'температура': 'прохладно', 'влажность': 'нормально', 'время_суток': 'вечер'}, ('мощность', 'обогрев')),
        FuzzyRule(23, {'температура': 'комфортно', 'влажность': 'сухо', 'время_суток': 'вечер'}, ('мощность', 'выключено')),
        FuzzyRule(24, {'температура': 'комфортно', 'влажность': 'нормально', 'время_суток': 'вечер'}, ('мощность', 'выключено')),
        FuzzyRule(25, {'температура': 'тепло', 'влажность': 'сухо', 'время_суток': 'вечер'}, ('мощность', 'выключено')),
        FuzzyRule(26, {'температура': 'тепло', 'влажность': 'влажно', 'время_суток': 'вечер'}, ('мощность', 'охлаждение')),
        FuzzyRule(27, {'температура': 'холодно', 'влажность': 'сухо', 'время_суток': 'ночь'}, ('мощность', 'обогрев')),
        FuzzyRule(28, {'температура': 'холодно', 'влажность': 'нормально', 'время_суток': 'ночь'}, ('мощность', 'обогрев')),
        FuzzyRule(29, {'температура': 'холодно', 'влажность': 'влажно', 'время_суток': 'ночь'}, ('мощность', 'обогрев')),
        FuzzyRule(30, {'температура': 'комфортно', 'влажность': 'нормально', 'время_суток': 'ночь'}, ('мощность', 'выключено')),
        FuzzyRule(31, {'температура': 'тепло', 'влажность': 'нормально', 'время_суток': 'ночь'}, ('мощность', 'выключено')),
        FuzzyRule(32, {'температура': 'жарко', 'влажность': 'нормально', 'время_суток': 'ночь'}, ('мощность', 'охлаждение')),
        FuzzyRule(33, {'температура': 'жарко', 'влажность': 'влажно', 'время_суток': 'ночь'}, ('мощность', 'охлаждение')),
    ]
    
    for rule in rules:
        fis.add_rule(rule)
    
    return fis

def test_three_mechanisms(system, test_input):
    results = {}
    
    system.set_implication('mamdani')
    system.set_aggregation('max')
    system.set_tnorm('min')
    output1, agg1, rules1 = system.inference(test_input)
    results['Мамдани (min-max)'] = {'output': output1, 'aggregated': agg1, 'rules': rules1, 'method': 'Композиционный (Мамдани)'}
    
    system.set_implication('larsen')
    system.set_aggregation('max')
    system.set_tnorm('prod')
    output2, agg2, rules2 = system.inference(test_input)
    results['Ларсена (prod-max)'] = {'output': output2, 'aggregated': agg2, 'rules': rules2, 'method': 'Композиционный (Ларсена)'}
    
    system.set_tnorm('min')
    output3 = system.inference_tsk(test_input)
    results['TSK (уровни истинности)'] = {'output': output3, 'aggregated': None, 'rules': None, 'method': 'Уровни истинности предпосылок'}
    
    return results

def compare_implications(system, test_input):
    results = {}
    
    for impl_name in ['mamdani', 'larsen', 'lukasiewicz']:
        system.set_implication(impl_name)
        system.set_aggregation('max')
        system.set_tnorm('min')
        output, aggregated, rules = system.inference(test_input)
        results[impl_name] = {'output': output, 'aggregated': aggregated, 'rules': rules}
    
    return results


class MamdaniApproximator:
    def __init__(self):
        self.input_mfs = []
        self.output_mfs = []
        self.rules = []
    
    def add_input_mf(self, name, func):
        self.input_mfs.append((name, func))
    
    def add_output_mf(self, name, func, center):
        self.output_mfs.append((name, func, center))
    
    def add_rule(self, input_idx, output_idx):
        self.rules.append((input_idx, output_idx))
    
    def predict(self, x_value):
        input_degrees = [func(x_value) for _, func in self.input_mfs]
        output_activations = []
        
        for input_idx, output_idx in self.rules:
            activation = input_degrees[input_idx]
            if activation > 0:
                output_activations.append((output_idx, activation))
        
        if not output_activations:
            return 0
        
        numerator = 0
        denominator = 0
        for output_idx, activation in output_activations:
            _, _, center = self.output_mfs[output_idx]
            numerator += activation * center
            denominator += activation
        
        return numerator / (denominator + 1e-10)

class TSKApproximator:
    def __init__(self):
        self.input_mfs = []
        self.rules = []
    
    def add_input_mf(self, name, func):
        self.input_mfs.append((name, func))
    
    def add_rule(self, input_idx, a, b):
        """Добавляет правило TSK: y = a*x + b"""
        self.rules.append((input_idx, a, b))
    
    def predict(self, x_value):
        input_degrees = [func(x_value) for _, func in self.input_mfs]
        numerator = 0
        denominator = 0
        
        for input_idx, a, b in self.rules:
            activation = input_degrees[input_idx]
            if activation > 0:
                output_value = a * x_value + b
                numerator += activation * output_value
                denominator += activation
        
        return numerator / (denominator + 1e-10)
    
    def get_tangent_lines(self, x_range):
        """Возвращает касательные линии"""
        tangent_lines = []
        x = np.linspace(x_range[0], x_range[1], 100)
        
        for input_idx, a, b in self.rules:
            y = a * x + b
            tangent_lines.append((x, y, self.input_mfs[input_idx][0]))
        
        return tangent_lines

def create_function_models(func_type):
    if func_type == "sin(x)":
        x_range = (0, 2*np.pi)
        
        mamdani = MamdaniApproximator()
        mamdani.add_input_mf('Низкий', lambda x: trapmf(x, [0, 0, 0.5, 2]))
        mamdani.add_input_mf('Средний_низкий', lambda x: trimf(x, [1, 2.5, 4]))
        mamdani.add_input_mf('Средний_высокий', lambda x: trimf(x, [3, 4.5, 6]))
        mamdani.add_input_mf('Высокий', lambda x: trapmf(x, [5, 6, 2*np.pi, 2*np.pi]))
        
        mamdani.add_output_mf('Полож_высокий', None, 1.0)
        mamdani.add_output_mf('Полож_низкий', None, 0.3)
        mamdani.add_output_mf('Отриц_низкий', None, -0.3)
        mamdani.add_output_mf('Отриц_высокий', None, -1.0)
        
        mamdani.add_rule(0, 0)
        mamdani.add_rule(1, 3)
        mamdani.add_rule(2, 3)
        mamdani.add_rule(3, 1)
        
        tsk = TSKApproximator()
        tsk.add_input_mf('Низкий', lambda x: trapmf(x, [0, 0, 0.5, 2]))
        tsk.add_input_mf('Средний_низкий', lambda x: trimf(x, [1, 2.5, 4]))
        tsk.add_input_mf('Средний_высокий', lambda x: trimf(x, [3, 4.5, 6]))
        tsk.add_input_mf('Высокий', lambda x: trapmf(x, [5, 6, 2*np.pi, 2*np.pi]))
        
        tsk.add_rule(0, 0.8, 0.0)
        tsk.add_rule(1, -0.5, 1.5)
        tsk.add_rule(2, -0.3, -0.5)
        tsk.add_rule(3, 0.2, -1.0)
        
    elif func_type == "x²":
        x_range = (-3, 3)
        
        mamdani = MamdaniApproximator()
        mamdani.add_input_mf('Очень_низкий', lambda x: trapmf(x, [-3, -3, -2.5, -1.5]))
        mamdani.add_input_mf('Низкий', lambda x: trimf(x, [-2, -1, 0]))
        mamdani.add_input_mf('Средний', lambda x: trimf(x, [-0.5, 0, 0.5]))
        mamdani.add_input_mf('Высокий', lambda x: trimf(x, [0, 1, 2]))
        mamdani.add_input_mf('Очень_высокий', lambda x: trapmf(x, [1.5, 2.5, 3, 3]))
        
        mamdani.add_output_mf('Малый', None, 0.2)
        mamdani.add_output_mf('Средний', None, 2.0)
        mamdani.add_output_mf('Большой', None, 6.0)
        
        mamdani.add_rule(0, 2)
        mamdani.add_rule(1, 1)
        mamdani.add_rule(2, 0)
        mamdani.add_rule(3, 1)
        mamdani.add_rule(4, 2)
        
        tsk = TSKApproximator()
        tsk.add_input_mf('Очень_низкий', lambda x: trapmf(x, [-3, -3, -2.5, -1.5]))
        tsk.add_input_mf('Низкий', lambda x: trimf(x, [-2, -1, 0]))
        tsk.add_input_mf('Средний', lambda x: trimf(x, [-0.5, 0, 0.5]))
        tsk.add_input_mf('Высокий', lambda x: trimf(x, [0, 1, 2]))
        tsk.add_input_mf('Очень_высокий', lambda x: trapmf(x, [1.5, 2.5, 3, 3]))
        
        tsk.add_rule(0, -4.0, 4.0)
        tsk.add_rule(1, -2.0, 1.0)
        tsk.add_rule(2, 0.5, 0.0)
        tsk.add_rule(3, 2.0, 1.0)
        tsk.add_rule(4, 4.0, 4.0)
    
    elif func_type == "e^(-x²)":
        x_range = (-2, 2)
        
        mamdani = MamdaniApproximator()
        mamdani.add_input_mf('Низкий', lambda x: trapmf(x, [-2, -2, -1.5, -0.5]))
        mamdani.add_input_mf('Средний', lambda x: trimf(x, [-1, 0, 1]))
        mamdani.add_input_mf('Высокий', lambda x: trapmf(x, [0.5, 1.5, 2, 2]))
        
        mamdani.add_output_mf('Малый', None, 0.2)
        mamdani.add_output_mf('Высокий', None, 1.0)
        
        mamdani.add_rule(0, 0)
        mamdani.add_rule(1, 1)
        mamdani.add_rule(2, 0)
        
        tsk = TSKApproximator()
        tsk.add_input_mf('Низкий', lambda x: trapmf(x, [-2, -2, -1.5, -0.5]))
        tsk.add_input_mf('Средний', lambda x: trimf(x, [-1, 0, 1]))
        tsk.add_input_mf('Высокий', lambda x: trapmf(x, [0.5, 1.5, 2, 2]))
        
        tsk.add_rule(0, 0.1, 0.3)
        tsk.add_rule(1, 0.0, 0.95)
        tsk.add_rule(2, -0.1, 0.3)
    
    return mamdani, tsk, x_range

def create_separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setStyleSheet("background-color: #CCCCCC;")
    return line

def create_styled_button(text, color):
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {color};
            color: white;
            padding: 12px 20px;
            font-size: 13px;
            font-weight: bold;
            border: none;
            border-radius: 6px;
        }}
        QPushButton:hover {{
            background-color: {color};
            opacity: 0.8;
        }}
        QPushButton:pressed {{
            background-color: {color};
            padding-top: 14px;
            padding-bottom: 10px;
        }}
    """)
    btn.setMinimumHeight(45)
    btn.setCursor(Qt.PointingHandCursor)
    return btn

class PlotWindow(QWidget):
    def __init__(self, aggregated, universe, crisp_output, output_name):
        super().__init__()
        self.setWindowTitle("График агрегированной функции принадлежности")
        self.setGeometry(100, 100, 950, 650)
        
        layout = QVBoxLayout()
        
        fig = Figure(figsize=(9, 5.5))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        ax.fill_between(universe, 0, aggregated, alpha=0.35, color='#2196F3', label='Агрегированная ФП')
        ax.plot(universe, aggregated, 'b-', linewidth=2.5)
        ax.axvline(x=crisp_output, color='#F44336', linestyle='--', linewidth=3, label=f'Дефаззификация: {crisp_output:.2f}')
        
        ax.set_xlabel('Значение', fontsize=13, fontweight='bold')
        ax.set_ylabel('Степень принадлежности', fontsize=13, fontweight='bold')
        ax.set_title(f'Результат нечёткого вывода: {output_name}', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(fontsize=12, loc='best')
        
        layout.addWidget(canvas)
        self.setLayout(layout)

class Part1Tab(QWidget):
    def __init__(self):
        super().__init__()
        self.lighting_system = create_lighting_control_system()
        self.climate_system_3 = create_climate_control_system_3()
        self.current_system = self.lighting_system
        
        self.input_sliders = {}
        self.input_labels = {}
        
        self.impl_map = {'Мамдани (min)': 'mamdani', 'Ларсена (prod)': 'larsen', 'Лукасевича': 'lukasiewicz'}
        self.aggr_map = {'Максимум': 'max', 'Вероятностная сумма': 'probsum'}
        self.tnorm_map = {'Минимум': 'min', 'Произведение': 'prod'}
        
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        info_label = QLabel("Часть 1: Проектирование систем управления умным домом")
        info_label.setFont(QFont("Arial", 12, QFont.Bold))
        info_label.setStyleSheet("background-color: #E3F2FD; padding: 10px; border-radius: 5px; color: #1976D2;")
        left_panel.addWidget(info_label)
        
        system_group = QGroupBox("Выбор системы")
        system_group.setFont(QFont("Arial", 11, QFont.Bold))
        system_layout = QVBoxLayout()
        
        self.radio_lighting = QRadioButton("Освещение (1 входная переменная)")
        self.radio_lighting.setFont(QFont("Arial", 10))
        self.radio_lighting.setChecked(True)
        self.radio_lighting.toggled.connect(self.switch_system)
        
        self.radio_climate = QRadioButton("Климат-контроль (3 входные переменные)")
        self.radio_climate.setFont(QFont("Arial", 10))
        self.radio_climate.toggled.connect(self.switch_system)
        
        system_layout.addWidget(self.radio_lighting)
        system_layout.addWidget(self.radio_climate)
        system_group.setLayout(system_layout)
        left_panel.addWidget(system_group)
        
        settings_group = QGroupBox("Параметры нечёткого вывода")
        settings_group.setFont(QFont("Arial", 11, QFont.Bold))
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(8)
        
        impl_layout = QHBoxLayout()
        impl_layout.addWidget(QLabel("Импликация:"))
        self.impl_combo = QComboBox()
        self.impl_combo.addItems(list(self.impl_map.keys()))
        self.impl_combo.setFont(QFont("Arial", 10))
        impl_layout.addWidget(self.impl_combo)
        settings_layout.addLayout(impl_layout)
        
        aggr_layout = QHBoxLayout()
        aggr_layout.addWidget(QLabel("Агрегация:"))
        self.aggr_combo = QComboBox()
        self.aggr_combo.addItems(list(self.aggr_map.keys()))
        self.aggr_combo.setFont(QFont("Arial", 10))
        aggr_layout.addWidget(self.aggr_combo)
        settings_layout.addLayout(aggr_layout)
        
        tnorm_layout = QHBoxLayout()
        tnorm_layout.addWidget(QLabel("T-норма:"))
        self.tnorm_combo = QComboBox()
        self.tnorm_combo.addItems(list(self.tnorm_map.keys()))
        self.tnorm_combo.setFont(QFont("Arial", 10))
        tnorm_layout.addWidget(self.tnorm_combo)
        settings_layout.addLayout(tnorm_layout)
        
        settings_group.setLayout(settings_layout)
        left_panel.addWidget(settings_group)
        
        self.input_group = QGroupBox("Входные параметры")
        self.input_group.setFont(QFont("Arial", 11, QFont.Bold))
        self.input_layout = QVBoxLayout()
        self.input_group.setLayout(self.input_layout)
        left_panel.addWidget(self.input_group)
        self.create_input_widgets()
        
        left_panel.addWidget(create_separator())
        
        calc_btn = create_styled_button("Вычислить", "#4CAF50")
        calc_btn.clicked.connect(self.calculate)
        left_panel.addWidget(calc_btn)
        
        test_btn = create_styled_button("Тест 3-х механизмов", "#2196F3")
        test_btn.clicked.connect(self.test_mechanisms)
        left_panel.addWidget(test_btn)
        
        compare_btn = create_styled_button("Сравнить импликации", "#FF9800")
        compare_btn.clicked.connect(self.compare_all_implications)
        left_panel.addWidget(compare_btn)
        
        save_btn = create_styled_button("Сохранить", "#9C27B0")
        save_btn.clicked.connect(self.save_system)
        left_panel.addWidget(save_btn)
        
        left_panel.addStretch()
        
        right_panel = QVBoxLayout()
        
        results_group = QGroupBox("Результаты")
        results_group.setFont(QFont("Arial", 11, QFont.Bold))
        results_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 10))
        self.result_text.setStyleSheet("background-color: #F5F5F5; border: 1px solid #CCCCCC; border-radius: 4px; padding: 8px;")
        results_layout.addWidget(self.result_text)
        
        results_group.setLayout(results_layout)
        right_panel.addWidget(results_group)
        
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(500)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
    
    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())
    
    def create_input_widgets(self):
        self.clear_layout(self.input_layout)
        self.input_sliders.clear()
        self.input_labels.clear()
        
        if self.radio_lighting.isChecked():
            self.add_slider("освещённость", 0, 100, 25, "люкс")
        else:
            self.add_slider("температура", 10, 35, 22, "°C")
            self.add_slider("влажность", 20, 80, 50, "%")
            self.add_slider("время_суток", 0, 23, 14, "час")
    
    def add_slider(self, name, min_val, max_val, default, unit):
        v_layout = QVBoxLayout()
        v_layout.setSpacing(5)
        
        label_name = QLabel(f"{name.replace('_', ' ').capitalize()}:")
        label_name.setFont(QFont("Arial", 10, QFont.Bold))
        v_layout.addWidget(label_name)
        
        h_layout = QHBoxLayout()
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * 10))
        slider.setMaximum(int(max_val * 10))
        slider.setValue(int(default * 10))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(int((max_val - min_val) * 10 / 5))
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #C4C4C4);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2196F3, stop:1 #1976D2);
                border: 1px solid #1565C0;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #42A5F5, stop:1 #1E88E5);
            }
        """)
        h_layout.addWidget(slider)
        
        value_label = QLabel(f"{default:.1f} {unit}")
        value_label.setMinimumWidth(80)
        value_label.setFont(QFont("Arial", 10, QFont.Bold))
        value_label.setStyleSheet("color: #1976D2; background-color: #E3F2FD; padding: 5px; border-radius: 3px;")
        h_layout.addWidget(value_label)
        
        slider.valueChanged.connect(lambda v, l=value_label, u=unit: l.setText(f"{v/10:.1f} {u}"))
        
        v_layout.addLayout(h_layout)
        
        self.input_sliders[name] = slider
        self.input_labels[name] = value_label
        
        self.input_layout.addLayout(v_layout)
    
    def switch_system(self):
        if self.radio_lighting.isChecked():
            self.current_system = self.lighting_system
        else:
            self.current_system = self.climate_system_3
        self.create_input_widgets()
        self.result_text.clear()
    
    def get_inputs(self):
        return {name: slider.value() / 10.0 for name, slider in self.input_sliders.items()}
    
    def get_english_values(self):
        return {
            'impl': self.impl_map[self.impl_combo.currentText()],
            'aggr': self.aggr_map[self.aggr_combo.currentText()],
            'tnorm': self.tnorm_map[self.tnorm_combo.currentText()]
        }
    
    def calculate(self):
        try:
            inputs = self.get_inputs()
            params = self.get_english_values()
            
            self.current_system.set_implication(params['impl'])
            self.current_system.set_aggregation(params['aggr'])
            self.current_system.set_tnorm(params['tnorm'])
            
            output, aggregated, rules = self.current_system.inference(inputs)
            
            result = "=" * 75 + "\n"
            result += f"  РЕЗУЛЬТАТЫ НЕЧЁТКОГО ВЫВОДА\n"
            result += "=" * 75 + "\n\n"
            result += f"Система: {self.current_system.name}\n"
            result += f"Импликация: {self.impl_combo.currentText()}\n"
            result += f"Агрегация: {self.aggr_combo.currentText()}\n"
            result += f"T-норма: {self.tnorm_combo.currentText()}\n\n"
            result += f"Входные данные:\n"
            for name, value in inputs.items():
                result += f"  - {name.capitalize()}: {value:.1f}\n"
            result += "\n" + "-" * 75 + "\n"
            result += "Активированные правила:\n\n"
            
            active_count = 0
            for rule, deg, _ in rules:
                if deg > 0.01:
                    active_count += 1
                    result += f"  [{active_count}] IF "
                    
                    conditions = []
                    for var, term in rule.antecedents.items():
                        conditions.append(f"{var} = '{term}'")
                    result += " AND ".join(conditions)
                    
                    out_var, out_term = rule.consequent
                    result += f"\n      THEN {out_var} = '{out_term}'\n"
                    
                    bar_length = int(deg * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    result += f"      Активация: {deg:.4f} [{bar}]\n\n"
            
            if active_count == 0:
                result += "  Нет активных правил\n"
            
            out_name = list(self.current_system.output_variables.keys())[0]
            result += "-" * 75 + "\n"
            result += f"Всего активных правил: {active_count}\n"
            result += "\n" + "=" * 75 + "\n"
            result += f"  ВЫХОДНОЕ ЗНАЧЕНИЕ ({out_name}): {output:.2f}\n"
            result += "=" * 75 + "\n"
            
            self.result_text.setText(result)
            
            universe = self.current_system.output_variables[out_name].universe
            self.plot_window = PlotWindow(aggregated, universe, output, out_name)
            self.plot_window.show()
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка вычисления:\n{str(e)}")
    
    def test_mechanisms(self):
        try:
            inputs = self.get_inputs()
            
            if not self.radio_lighting.isChecked():
                QMessageBox.warning(self, "Внимание",
                    "Тест 3-х механизмов работает только для системы освещения (1 вход).")
                return
            
            results = test_three_mechanisms(self.lighting_system, inputs)
            
            output = "=" * 85 + "\n"
            output += "  ТЕСТИРОВАНИЕ ТРЁХ МЕХАНИЗМОВ (Пункт 2)\n"
            output += "=" * 85 + "\n\n"
            output += f"Тестовые входные данные:\n"
            for name, value in inputs.items():
                output += f"  - {name.capitalize()}: {value:.1f}\n"
            output += "\n"
            
            for mech_name, data in results.items():
                output += "=" * 85 + "\n"
                output += f"МЕХАНИЗМ: {mech_name}\n"
                output += f"Описание: {data['method']}\n"
                output += "=" * 85 + "\n\n"
                
                if data['rules']:
                    output += "Активированные правила:\n\n"
                    active_count = 0
                    for rule, degree, _ in data['rules']:
                        if degree > 0.01:
                            active_count += 1
                            output += f"  [{active_count}] IF "
                            
                            conditions = []
                            for var, term in rule.antecedents.items():
                                conditions.append(f"{var} = '{term}'")
                            output += " AND ".join(conditions)
                            
                            out_var, out_term = rule.consequent
                            output += f"\n      THEN {out_var} = '{out_term}'\n"
                            
                            bar_length = int(degree * 30)
                            bar = "█" * bar_length + "░" * (30 - bar_length)
                            output += f"      Активация: {degree:.4f} [{bar}]\n\n"
                    
                    output += f"Всего активных правил: {active_count}\n"
                
                output += f"\n{'─' * 85}\n"
                output += f"  РЕЗУЛЬТАТ: {data['output']:.4f}\n"
                output += f"{'─' * 85}\n\n"
            
            output += "=" * 85 + "\n"
            output += "  СРАВНИТЕЛЬНАЯ ТАБЛИЦА\n"
            output += "=" * 85 + "\n"
            output += f"{'Механизм':<45} | {'Результат':>12}\n"
            output += "-" * 85 + "\n"
            for name, data in results.items():
                output += f"{name:<45} | {data['output']:>12.4f}\n"
            output += "=" * 85 + "\n"
            
            self.result_text.setText(output)
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка тестирования:\n{str(e)}")
    
    def compare_all_implications(self):
        try:
            inputs = self.get_inputs()
            results = compare_implications(self.current_system, inputs)
            
            impl_names_ru = {'mamdani': 'Мамдани', 'larsen': 'Ларсена', 'lukasiewicz': 'Лукасевича'}
            
            output = "=" * 85 + "\n"
            output += "  СРАВНЕНИЕ ТИПОВ ИМПЛИКАЦИЙ (Пункт 4)\n"
            output += "=" * 85 + "\n\n"
            output += f"Тестовые входные данные:\n"
            for name, value in inputs.items():
                output += f"  - {name.capitalize()}: {value:.1f}\n"
            output += "\n"
            
            for impl_name, data in results.items():
                rus_name = impl_names_ru.get(impl_name, impl_name)
                output += "=" * 85 + "\n"
                output += f"ИМПЛИКАЦИЯ: {rus_name}\n"
                output += "=" * 85 + "\n\n"
                
                output += "Активированные правила:\n\n"
                active_count = 0
                for rule, degree, _ in data['rules']:
                    if degree > 0.01:
                        active_count += 1
                        output += f"  [{active_count}] IF "
                        
                        conditions = []
                        for var, term in rule.antecedents.items():
                            conditions.append(f"{var} = '{term}'")
                        output += " AND ".join(conditions)
                        
                        out_var, out_term = rule.consequent
                        output += f"\n      THEN {out_var} = '{out_term}'\n"
                        
                        bar_length = int(degree * 30)
                        bar = "█" * bar_length + "░" * (30 - bar_length)
                        output += f"      Активация: {degree:.4f} [{bar}]\n\n"
                
                output += f"Всего активных правил: {active_count}\n"
                output += f"\n{'─' * 85}\n"
                output += f"  РЕЗУЛЬТАТ: {data['output']:.4f}\n"
                output += f"{'─' * 85}\n\n"
            
            output += "=" * 85 + "\n"
            output += "  СРАВНИТЕЛЬНАЯ ТАБЛИЦА\n"
            output += "=" * 85 + "\n"
            output += f"{'Тип импликации':<25} | {'Результат':>15} | {'Активных правил':>18}\n"
            output += "-" * 85 + "\n"
            
            for impl_name, data in results.items():
                active_rules = sum(1 for _, deg, _ in data['rules'] if deg > 0.01)
                rus_name = impl_names_ru.get(impl_name, impl_name)
                output += f"{rus_name:<25} | {data['output']:>15.4f} | {active_rules:>18}\n"
            
            output += "=" * 85 + "\n"
            
            values = [data['output'] for data in results.values()]
            max_val, min_val = max(values), min(values)
            diff = max_val - min_val
            
            output += "\n  АНАЛИЗ РЕЗУЛЬТАТОВ:\n"
            output += f"  - Максимальное значение: {max_val:.4f}\n"
            output += f"  - Минимальное значение: {min_val:.4f}\n"
            output += f"  - Разброс: {diff:.4f} ({(diff/max_val*100):.2f}%)\n"
            output += "=" * 85 + "\n"
            
            self.result_text.setText(output)
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сравнения:\n{str(e)}")
    
    def save_system(self):
        try:
            filename = f"{self.current_system.name}.json"
            self.current_system.save_to_file(filename)
            QMessageBox.information(self, "Успешно", f"Система сохранена:\n\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения:\n{str(e)}")

class Part2Tab(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        info_label = QLabel("Часть 2: Моделирование нелинейных функций")
        info_label.setFont(QFont("Arial", 12, QFont.Bold))
        info_label.setStyleSheet("background-color: #FFF3E0; padding: 10px; border-radius: 5px; color: #E65100;")
        left_panel.addWidget(info_label)
        
        func_group = QGroupBox("Параметры моделирования")
        func_group.setFont(QFont("Arial", 11, QFont.Bold))
        func_layout = QVBoxLayout()
        
        func_select_layout = QHBoxLayout()
        func_select_layout.addWidget(QLabel("Выберите функцию:"))
        self.func_combo = QComboBox()
        self.func_combo.addItems(["sin(x)", "x²", "e^(-x²)"])
        self.func_combo.setFont(QFont("Arial", 10))
        func_select_layout.addWidget(self.func_combo)
        func_select_layout.addStretch()
        func_layout.addLayout(func_select_layout)
        
        rules_info = QLabel("TSK использует линейные функции y=ax+b")
        rules_info.setFont(QFont("Arial", 9))
        rules_info.setStyleSheet("color: #666666; padding: 5px; background-color: #F0F0F0; border-radius: 3px;")
        func_layout.addWidget(rules_info)
        
        func_group.setLayout(func_layout)
        left_panel.addWidget(func_group)
        
        model_btn = create_styled_button("Построить модели и сравнить", "#FF5722")
        model_btn.setMinimumHeight(55)
        model_btn.clicked.connect(self.model_function)
        left_panel.addWidget(model_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        left_panel.addWidget(self.progress_bar)
        
        left_panel.addWidget(create_separator())
        
        results_group = QGroupBox("Результаты моделирования")
        results_group.setFont(QFont("Arial", 11, QFont.Bold))
        results_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 9))
        self.result_text.setStyleSheet("background-color: #F5F5F5; border: 1px solid #CCCCCC; border-radius: 4px; padding: 8px;")
        results_layout.addWidget(self.result_text)
        
        results_group.setLayout(results_layout)
        left_panel.addWidget(results_group)
        
        right_panel = QVBoxLayout()
        
        graph_group = QGroupBox("Визуализация")
        graph_group.setFont(QFont("Arial", 11, QFont.Bold))
        graph_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        graph_layout.addWidget(self.canvas)
        
        graph_group.setLayout(graph_layout)
        right_panel.addWidget(graph_group)
        
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(500)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
    
    def get_true_function(self, func_type, x):
        if func_type == "sin(x)":
            return np.sin(x)
        elif func_type == "x²":
            return x**2
        elif func_type == "e^(-x²)":
            return np.exp(-x**2)
    
    def model_function(self):
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(20)
            QApplication.processEvents()
            
            func_type = self.func_combo.currentText()
            
            self.progress_bar.setValue(40)
            QApplication.processEvents()
            
            mamdani, tsk, x_range = create_function_models(func_type)
            
            self.progress_bar.setValue(60)
            QApplication.processEvents()
            
            x = np.linspace(x_range[0], x_range[1], 100)
            y_true = self.get_true_function(func_type, x)
            y_mamdani = np.array([mamdani.predict(xi) for xi in x])
            y_tsk = np.array([tsk.predict(xi) for xi in x])
            
            self.progress_bar.setValue(80)
            QApplication.processEvents()
            
            mae_mamdani = np.mean(np.abs(y_true - y_mamdani))
            mae_tsk = np.mean(np.abs(y_true - y_tsk))
            mse_mamdani = np.mean((y_true - y_mamdani)**2)
            mse_tsk = np.mean((y_true - y_tsk)**2)
            
            result = "=" * 70 + "\n"
            result += f"  МОДЕЛИРОВАНИЕ: {func_type}\n"
            result += "=" * 70 + "\n\n"
            result += f"Параметры:\n"
            result += f"  - Функция: y = {func_type}\n"
            result += f"  - Интервал: [{x_range[0]:.2f}, {x_range[1]:.2f}]\n"
            result += f"  - Правил (Мамдани): {len(mamdani.rules)}\n"
            result += f"  - Правил (TSK): {len(tsk.rules)}\n\n"
            
            result += "-" * 70 + "\n"
            result += "ЛИНЕЙНЫЕ ФУНКЦИИ TSK (y = a*x + b):\n"
            for i, (idx, a, b) in enumerate(tsk.rules, 1):
                area_name = tsk.input_mfs[idx][0]
                result += f"  Правило {i} ({area_name}):\n"
                result += f"    y = {a:.2f}*x + ({b:.2f})\n"
            
            result += "\n" + "-" * 70 + "\n"
            result += "МОДЕЛЬ МАМДАНИ:\n"
            result += f"  - MAE: {mae_mamdani:.6f}\n"
            result += f"  - MSE: {mse_mamdani:.6f}\n\n"
            result += "МОДЕЛЬ ТАКАГИ-СУГЕНО:\n"
            result += f"  - MAE: {mae_tsk:.6f}\n"
            result += f"  - MSE: {mse_tsk:.6f}\n\n"
            result += "=" * 70 + "\n"
            
            if mae_tsk < mae_mamdani:
                improvement = ((mae_mamdani - mae_tsk) / mae_mamdani * 100)
                result += f"ВЫВОД: TSK точнее на {improvement:.2f}%\n"
            else:
                improvement = ((mae_tsk - mae_mamdani) / mae_tsk * 100)
                result += f"ВЫВОД: Мамдани точнее на {improvement:.2f}%\n"
            
            result += "=" * 70 + "\n"
            
            self.result_text.setText(result)
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            ax.plot(x, y_true, 'b-', linewidth=3, label=f'Исходная функция {func_type}', alpha=0.8)
            ax.plot(x, y_mamdani, 'r--', linewidth=2.5, label='Модель Мамдани', alpha=0.9)
            ax.plot(x, y_tsk, 'g-.', linewidth=2.5, label='Модель Такаги-Сугено', alpha=0.9)
            
            tangent_lines = tsk.get_tangent_lines(x_range)
            colors = ['orange', 'purple', 'brown', 'pink', 'cyan']
            
            for i, (x_line, y_line, name) in enumerate(tangent_lines):
                idx, a, b = tsk.rules[i]
                color = colors[i % len(colors)]
                
                ax.plot(x_line, y_line, ':', color=color, linewidth=1.5, 
                    alpha=0.6, label=f'Линейная {i+1}: {name}')
                
                mid_idx = len(x_line) // 2
                x_text = x_line[mid_idx]
                y_text = y_line[mid_idx]
                
                if b >= 0:
                    label_text = f'y={a:.2f}x+{b:.2f}'
                else:
                    label_text = f'y={a:.2f}x{b:.2f}'
                
                ax.annotate(label_text, 
                        xy=(x_text, y_text),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', 
                                    edgecolor=color, 
                                    alpha=0.8),
                        ha='left')
            
            ax.set_xlabel('x', fontsize=13, fontweight='bold')
            ax.set_ylabel('y', fontsize=13, fontweight='bold')
            ax.set_title(f'Аппроксимация {func_type} с линейными функциями TSK', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=10, loc='best', framealpha=0.9, ncol=2)
            ax.grid(True, alpha=0.25, linestyle='--')
            
            self.canvas.draw()
            
            self.progress_bar.setValue(100)
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Ошибка", f"Ошибка:\n{str(e)}")


class FuzzyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа №3: Нечёткие модели")
        self.setGeometry(80, 50, 1450, 950)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FAFAFA;
            }
            QTabWidget::pane {
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #E0E0E0;
                padding: 12px 25px;
                margin: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 3px solid #2196F3;
            }
            QTabBar::tab:hover {
                background-color: #F5F5F5;
            }
            QGroupBox {
                border: 2px solid #DDDDDD;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 5px 10px;
                background-color: white;
            }
        """)
        
        self.setup_ui()
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        title = QLabel("Лабораторная работа №3: Нечёткие модели")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #2196F3, stop:0.5 #21CBF3, stop:1 #2196F3);
            color: white;
            padding: 20px;
            border-radius: 8px;
        """)
        main_layout.addWidget(title)
        
        tabs = QTabWidget()
        tabs.setFont(QFont("Arial", 11))
        tabs.addTab(Part1Tab(), "Часть 1: Управление умным домом")
        tabs.addTab(Part2Tab(), "Часть 2: Моделирование функций")
        
        main_layout.addWidget(tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(250, 250, 250))
    palette.setColor(QPalette.WindowText, Qt.black)
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.black)
    palette.setColor(QPalette.Text, Qt.black)
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, Qt.black)
    palette.setColor(QPalette.Highlight, QColor(33, 150, 243))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)
    
    window = FuzzyGUI()
    window.show()
    sys.exit(app.exec_())
