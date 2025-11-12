import re
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

from colors import Colors


class BackwardExpertSystem:
    """
    Экспертная система с обратной цепочкой рассуждений
    для предметной области "Умный дом"
    """

    def __init__(self, rules_file: str = "rules.txt"):
        self.rules_file = rules_file
        self.rules = []
        self.facts = {}
        self.asked_facts = set()
        self.inference_log = []
        self.animation_speed = 0.05
        self.recursion_depth = 0
        self.max_depth = 50
        self.load_rules()
        self.initialize_facts()

    def clear_screen(self):
        """Очистка экрана"""
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self, text: str, color=Colors.BRIGHT_CYAN):
        """Печать заголовка с рамкой"""
        width = 60
        border = "═" * width
        print(f"\n{color}╔{border}╗{Colors.RESET}")
        print(f"{color}║{text.center(width)}║{Colors.RESET}")
        print(f"{color}╚{border}╝{Colors.RESET}")

    def print_section(self, title: str, color=Colors.BRIGHT_YELLOW):
        """Печать секции"""
        print(f"\n{color}▓▓▓ {title} ▓▓▓{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * (len(title) + 8)}{Colors.RESET}")

    def print_success(self, message: str):
        """Печать сообщения об успехе"""
        print(f"{Colors.BRIGHT_GREEN}✓ {message}{Colors.RESET}")

    def print_warning(self, message: str):
        """Печать предупреждения"""
        print(f"{Colors.BRIGHT_YELLOW}⚠ {message}{Colors.RESET}")

    def print_error(self, message: str):
        """Печать ошибки"""
        print(f"{Colors.BRIGHT_RED}✗ {message}{Colors.RESET}")

    def print_info(self, message: str):
        """Печать информации"""
        print(f"{Colors.BRIGHT_BLUE}ℹ {message}{Colors.RESET}")

    def print_fact(self, key: str, value: str):
        """Печать факта с цветовой разметкой"""
        print(f"  📋 {Colors.CYAN}{key}{Colors.RESET} = {Colors.BRIGHT_WHITE}{value}{Colors.RESET}")

    def animate_text(self, text: str, delay: float = None):
        """Анимированный вывод текста"""
        if delay is None:
            delay = self.animation_speed
        for char in text:
            print(char, end="", flush=True)
            time.sleep(delay)
        print()

    def print_depth_indent(self):
        """Печать отступа в зависимости от глубины рекурсии"""
        return "  " * self.recursion_depth

    def initialize_facts(self):
        """Инициализация стартовой ситуации"""
        self.facts = {
            "время_суток": "вечер",
            "день_недели": "рабочий",
            "присутствие_людей": "да",
            "температура_внешняя": "холодно",
            "освещенность": "темно",
        }

        self.print_section("Инициализация системы", Colors.BRIGHT_MAGENTA)
        self.animate_text("🏠 Загружаю параметры умного дома...")
        time.sleep(0.5)

        print(f"\n{Colors.BRIGHT_CYAN}📋 Известные факты:{Colors.RESET}")
        for key, value in self.facts.items():
            self.print_fact(key, value)

        print(f"\n{Colors.DIM}{'─' * 50}{Colors.RESET}")
        self.asked_facts.clear()
        self.inference_log.clear()

    def load_rules(self):
        """Загрузка правил из файла"""
        try:
            with open(self.rules_file, "r", encoding="utf-8") as f:
                self.rules = []
                rule_count = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            rule = self.parse_rule(line)
                            if rule:
                                self.rules.append(rule)
                                rule_count += 1
                        except Exception as e:
                            self.print_error(f"Ошибка в строке {line_num}: {e}")

                if rule_count > 0:
                    self.print_success(f"Загружено правил: {rule_count}")
                else:
                    self.print_warning("Правила не найдены")

        except FileNotFoundError:
            self.print_warning(f"Файл {self.rules_file} не найден")
            self.animate_text("📝 Создаю новый файл с базовыми правилами...")
            self.create_default_rules()

    def create_default_rules(self):
        """Создание файла с базовыми правилами для умного дома"""
        default_rules = [
            "# Правила для системы освещения",
            "ЕСЛИ время_суток=вечер И присутствие_людей=да ТО включить_основное_освещение=да",
            "ЕСЛИ время_суток=ночь И движение_в_коридоре=да ТО включить_ночник=да",
            "ЕСЛИ освещенность=темно И присутствие_людей=да ТО включить_основное_освещение=да",
            "ЕСЛИ присутствие_людей=нет ТО выключить_все_освещение=да",
            "",
            "# Правила для системы отопления",
            "ЕСЛИ температура_внешняя=холодно И присутствие_людей=да ТО включить_отопление=да",
            "ЕСЛИ температура_внутренняя=жарко ТО уменьшить_отопление=да",
            "ЕСЛИ присутствие_людей=нет И день_недели=рабочий ТО режим_экономии_тепла=да",
            "",
            "# Правила безопасности",
            "ЕСЛИ присутствие_людей=нет И время_суток=день ТО включить_охрану=да",
            "ЕСЛИ движение_на_входе=да И включить_охрану=да ТО сигнал_тревоги=да",
            "ЕСЛИ дым=да ТО пожарная_тревога=да",
            "ЕСЛИ утечка_газа=да ТО перекрыть_газ=да",
            "",
            "# Правила для развлечений",
            "ЕСЛИ время_суток=вечер И день_недели=выходной И присутствие_людей=да ТО включить_развлекательную_систему=да",
            "ЕСЛИ включить_развлекательную_систему=да ТО приглушить_освещение=да",
            "",
            "# Правила для экономии энергии",
            "ЕСЛИ присутствие_людей=нет ТО режим_экономии_энергии=да",
            "ЕСЛИ режим_экономии_энергии=да ТО отключить_неприоритетные_устройства=да",
            "",
            "# Утренние правила",
            "ЕСЛИ время_суток=утро И день_недели=рабочий И присутствие_людей=да ТО включить_кофеварку=да",
            "ЕСЛИ время_суток=утро И присутствие_людей=да ТО включить_новости=да",
        ]

        with open(self.rules_file, "w", encoding="utf-8") as f:
            f.write("\n".join(default_rules))
        self.load_rules()

    def parse_rule(self, rule_text: str) -> Optional[Dict]:
        """Парсинг правила вида: ЕСЛИ условие ТО заключение"""
        rule_text = " ".join(rule_text.split())

        pattern = r"ЕСЛИ\s+(.+?)\s+ТО\s+(.+)"
        match = re.match(pattern, rule_text, re.IGNORECASE)

        if not match:
            return None

        conditions_str = match.group(1)
        conclusion_str = match.group(2)

        conditions = []
        condition_parts = re.split(r"\s+И\s+", conditions_str, flags=re.IGNORECASE)

        for part in condition_parts:
            cond_match = re.match(r"(\w+)\s*=\s*(.+)", part.strip())
            if cond_match:
                obj = cond_match.group(1).strip()
                value = cond_match.group(2).strip()
                conditions.append((obj, value))

        concl_match = re.match(r"(\w+)\s*=\s*(.+)", conclusion_str.strip())
        if not concl_match:
            return None

        conclusion_obj = concl_match.group(1).strip()
        conclusion_value = concl_match.group(2).strip()

        return {
            "conditions": conditions,
            "conclusion": (conclusion_obj, conclusion_value),
            "text": rule_text,
        }

    def ask_user(self, fact_name: str) -> Optional[str]:
        """Запрос значения факта у пользователя"""
        if fact_name in self.asked_facts:
            return None

        self.asked_facts.add(fact_name)

        print(f"\n{self.print_depth_indent()}{Colors.BRIGHT_YELLOW}❓ Требуется информация:{Colors.RESET}")
        print(f"{self.print_depth_indent()}{Colors.CYAN}   {fact_name}{Colors.RESET}")

        possible_values = {
            "движение_в_коридоре": "да/нет",
            "движение_на_входе": "да/нет",
            "температура_внутренняя": "жарко/нормально/холодно",
            "дым": "да/нет",
            "утечка_газа": "да/нет",
            "время_суток": "утро/день/вечер/ночь",
            "день_недели": "рабочий/выходной",
            "освещенность": "светло/темно",
        }

        if fact_name in possible_values:
            print(
                f"{self.print_depth_indent()}{Colors.DIM}   Возможные значения: {possible_values[fact_name]}{Colors.RESET}"
            )

        user_input = input(
            f"{self.print_depth_indent()}{Colors.BRIGHT_WHITE}   ➤ Введите значение (или 'нет' для пропуска): {Colors.RESET}"
        ).strip()

        if user_input.lower() in ["нет", "no", "skip", ""]:
            print(f"{self.print_depth_indent()}{Colors.DIM}   ⊗ Факт пропущен{Colors.RESET}")
            return None

        print(
            f"{self.print_depth_indent()}{Colors.BRIGHT_GREEN}   ✓ Добавлено: {fact_name} = {user_input}{Colors.RESET}"
        )
        return user_input

    def backward_chaining(self, goal: Tuple[str, str], trace: bool = True) -> bool:
        """
        Обратная цепочка рассуждений
        Пытается доказать цель goal = (объект, значение)
        """
        self.recursion_depth += 1

        if self.recursion_depth > self.max_depth:
            self.recursion_depth -= 1
            self.print_warning("Достигнута максимальная глубина рекурсии")
            return False

        goal_obj, goal_value = goal

        if trace:
            print(
                f"\n{self.print_depth_indent()}{Colors.BRIGHT_MAGENTA}🎯 Цель: {Colors.CYAN}{goal_obj} = {goal_value}{Colors.RESET}"
            )

        if goal_obj in self.facts:
            result = self.facts[goal_obj] == goal_value
            if trace:
                if result:
                    print(
                        f"{self.print_depth_indent()}{Colors.BRIGHT_GREEN}✓ Найдено в базе фактов: {goal_obj} = {self.facts[goal_obj]}{Colors.RESET}"
                    )
                else:
                    print(
                        f"{self.print_depth_indent()}{Colors.BRIGHT_RED}✗ Противоречие: {goal_obj} = {self.facts[goal_obj]} ≠ {goal_value}{Colors.RESET}"
                    )
            self.recursion_depth -= 1
            return result

        applicable_rules = []
        for rule in self.rules:
            concl_obj, concl_value = rule["conclusion"]
            if concl_obj == goal_obj and concl_value == goal_value:
                applicable_rules.append(rule)

        if trace and applicable_rules:
            print(
                f"{self.print_depth_indent()}{Colors.BRIGHT_BLUE}📚 Найдено правил для проверки: {len(applicable_rules)}{Colors.RESET}"
            )

        for i, rule in enumerate(applicable_rules, 1):
            if trace:
                print(
                    f"\n{self.print_depth_indent()}{Colors.BRIGHT_YELLOW}🔍 Проверяю правило #{i}:{Colors.RESET}"
                )
                print(f"{self.print_depth_indent()}{Colors.DIM}   {rule['text']}{Colors.RESET}")

            all_conditions_proved = True

            for cond_obj, cond_value in rule["conditions"]:
                subgoal = (cond_obj, cond_value)

                if trace:
                    print(
                        f"{self.print_depth_indent()}{Colors.DIM}├─ Подцель: {cond_obj} = {cond_value}{Colors.RESET}"
                    )

                if not self.backward_chaining(subgoal, trace):
                    all_conditions_proved = False
                    if trace:
                        print(
                            f"{self.print_depth_indent()}{Colors.BRIGHT_RED}└─ ✗ Подцель не доказана{Colors.RESET}"
                        )
                    break

            if all_conditions_proved:
                self.facts[goal_obj] = goal_value

                log_entry = {
                    "rule": rule["text"],
                    "goal": f"{goal_obj} = {goal_value}",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
                self.inference_log.append(log_entry)

                if trace:
                    print(
                        f"\n{self.print_depth_indent()}{Colors.BRIGHT_GREEN}✓✓✓ ЦЕЛЬ ДОКАЗАНА: {goal_obj} = {goal_value}{Colors.RESET}"
                    )
                    print(
                        f"{self.print_depth_indent()}{Colors.BRIGHT_GREEN}    Использовано правило: {rule['text']}{Colors.RESET}"
                    )

                self.recursion_depth -= 1
                return True

        if trace:
            print(
                f"\n{self.print_depth_indent()}{Colors.BRIGHT_YELLOW}💭 Не удалось доказать через правила{Colors.RESET}"
            )

        user_value = self.ask_user(goal_obj)

        if user_value is not None:
            self.facts[goal_obj] = user_value
            result = user_value == goal_value

            if trace:
                if result:
                    print(
                        f"{self.print_depth_indent()}{Colors.BRIGHT_GREEN}✓ Цель подтверждена пользователем{Colors.RESET}"
                    )
                else:
                    print(
                        f"{self.print_depth_indent()}{Colors.BRIGHT_RED}✗ Цель опровергнута пользователем{Colors.RESET}"
                    )

            self.recursion_depth -= 1
            return result

        if trace:
            print(
                f"{self.print_depth_indent()}{Colors.BRIGHT_RED}✗✗✗ ЦЕЛЬ НЕ ДОКАЗАНА: {goal_obj} = {goal_value}{Colors.RESET}"
            )

        self.recursion_depth -= 1
        return False

    def show_inference_log(self):
        """Показать лог вывода"""
        if not self.inference_log:
            self.print_info("Лог пуст - правила еще не применялись")
            return

        self.print_section("Журнал логического вывода", Colors.BRIGHT_MAGENTA)

        for i, entry in enumerate(self.inference_log, 1):
            print(f"\n{Colors.BRIGHT_MAGENTA}#{i} [{entry['timestamp']}]{Colors.RESET}")
            print(f"{Colors.DIM}├─ Правило:{Colors.RESET} {entry['rule']}")
            print(
                f"{Colors.DIM}└─ Доказано:{Colors.RESET} {Colors.BRIGHT_GREEN}{entry['goal']}{Colors.RESET}"
            )

    def display_facts(self):
        """Вывод текущих фактов"""
        self.print_section("Текущие факты", Colors.BRIGHT_CYAN)
        if not self.facts:
            self.print_warning("Фактов нет")
            return

        for key, value in self.facts.items():
            self.print_fact(key, value)

    def run(self):
        """Основной цикл работы экспертной системы с обратной цепочкой"""
        self.clear_screen()
        self.print_header("ЭКСПЕРТНАЯ СИСТЕМА 'УМНЫЙ ДОМ' (ОБРАТНАЯ ЦЕПОЧКА)")

        self.print_section("Задание целевой ситуации", Colors.BRIGHT_YELLOW)

        print(f"\n{Colors.BRIGHT_BLUE}💡 Примеры целей:{Colors.RESET}")
        print(f"  • {Colors.CYAN}включить_основное_освещение=да{Colors.RESET}")
        print(f"  • {Colors.CYAN}включить_отопление=да{Colors.RESET}")
        print(f"  • {Colors.CYAN}режим_экономии_энергии=да{Colors.RESET}")
        print(f"  • {Colors.CYAN}сигнал_тревоги=да{Colors.RESET}")
        print(f"  • {Colors.CYAN}включить_кофеварку=да{Colors.RESET}")

        goal_input = input(
            f"\n{Colors.BRIGHT_WHITE}🎯 Введите целевую ситуацию (объект=значение): {Colors.RESET}"
        ).strip()

        if not goal_input:
            self.print_error("Цель не задана")
            return

        match = re.match(r"(\w+)\s*=\s*(.+)", goal_input)
        if not match:
            self.print_error("Неверный формат. Используйте: объект=значение")
            return

        goal_obj = match.group(1).strip()
        goal_value = match.group(2).strip()
        goal = (goal_obj, goal_value)

        self.print_section("Процесс доказательства", Colors.BRIGHT_BLUE)
        self.animate_text("🧠 Запускаю обратную цепочку рассуждений...")
        time.sleep(0.5)

        self.recursion_depth = 0
        result = self.backward_chaining(goal, trace=True)

        self.print_section("Результат", Colors.BRIGHT_GREEN if result else Colors.BRIGHT_RED)

        if result:
            self.print_success(f"ЦЕЛЬ ДОСТИГНУТА: {goal_obj} = {goal_value}")
            print(f"\n{Colors.BRIGHT_GREEN}🎉 Целевая ситуация доказана!{Colors.RESET}")
        else:
            self.print_error(f"ЦЕЛЬ НЕ ДОСТИГНУТА: {goal_obj} = {goal_value}")
            print(
                f"\n{Colors.BRIGHT_RED}❌ Целевая ситуация опровергнута или не может быть доказана{Colors.RESET}"
            )

        self.display_facts()

        if self.inference_log:
            self.show_inference_log()

        input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")
