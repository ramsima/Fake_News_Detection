# custom_filters.py
from django import template

register = template.Library()

@register.filter(name='zip_lists')
def zip_lists(a, b):
    return zip(a, b)

@register.filter
def multiply_and_round(value, args):
    try:
        multiplier, decimals = map(int, args.split(','))
        result = value * multiplier
        return round(result, decimals)
    except (TypeError, ValueError):
        return ''
