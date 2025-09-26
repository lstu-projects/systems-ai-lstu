import time

from colors import Colors
from expert_system import ExpertSystem


def main():
    """Главная функция"""
    system = ExpertSystem()

    while True:
        system.clear_screen()
        system.print_header("ГЛАВНОЕ МЕНЮ", Colors.BRIGHT_CYAN)

        print(f"\n{Colors.DIM}┌─ Статистика системы ─────────────────────────────┐{Colors.RESET}")
        print(
            f"{Colors.DIM}│{Colors.RESET} Правил в базе: {Colors.BRIGHT_YELLOW}{len(system.rules):<26}{Colors.DIM}│{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}│{Colors.RESET} Текущих фактов: {Colors.BRIGHT_GREEN}{len(system.facts):<25}{Colors.DIM}│{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}│{Colors.RESET} Выведено фактов: {Colors.BRIGHT_BLUE}{len(system.derived_facts):<24}{Colors.DIM}│{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}│{Colors.RESET} Файл правил: {Colors.BRIGHT_WHITE}{system.rules_file:<28}{Colors.DIM}│{Colors.RESET}"
        )
        print(f"{Colors.DIM}└──────────────────────────────────────────────────┘{Colors.RESET}")

        print(f"\n{Colors.BRIGHT_BLUE}🏠 Выберите действие:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_CYAN}1.{Colors.RESET} {Colors.BRIGHT_GREEN}🚀 Запустить экспертную систему{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_CYAN}2.{Colors.RESET} {Colors.BRIGHT_YELLOW}📝 Редактировать правила{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_CYAN}3.{Colors.RESET} {Colors.BRIGHT_BLUE}📋 Показать текущие факты{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_CYAN}4.{Colors.RESET} {Colors.BRIGHT_MAGENTA}🔄 Сбросить к стартовому состоянию{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_CYAN}5.{Colors.RESET} {Colors.BRIGHT_WHITE}📊 Показать журнал вывода{Colors.RESET}"
        )
        print(f"  {Colors.BRIGHT_CYAN}6.{Colors.RESET} {Colors.BRIGHT_RED}🚪 Выйти{Colors.RESET}")

        try:
            choice = input(f"\n{Colors.BRIGHT_WHITE}➤ Ваш выбор (1-6): {Colors.RESET}").strip()

            if choice == "1":
                system.run()
            elif choice == "2":
                system.edit_rules_menu()
            elif choice == "3":
                system.display_facts()
                input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")
            elif choice == "4":
                print(f"\n{Colors.BRIGHT_YELLOW}🔄 Сброс к стартовому состоянию...{Colors.RESET}")
                system.initialize_facts()
                system.print_success("Система сброшена к начальному состоянию")
                input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")
            elif choice == "5":
                system.show_inference_log()
                input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")
            elif choice == "6":
                system.print_section("До свидания!", Colors.BRIGHT_MAGENTA)
                system.animate_text("🏠 Благодарим за использование системы умного дома!")
                break
            else:
                system.print_error("Неверный выбор. Введите число от 1 до 6")
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n{Colors.BRIGHT_YELLOW}Работа прервана пользователем{Colors.RESET}")
            break
        except Exception as e:
            system.print_error(f"Произошла ошибка: {e}")
            input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")


if __name__ == "__main__":
    main()
