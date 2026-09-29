"""Checkbox values and events use the same contract on every panel."""
import unittest

from wrs.viewer.web_ui import UIManager, UIPanel


class CheckboxTests(unittest.TestCase):
    def setUp(self):
        self.ui = UIManager()
        self.panel = self.ui.add_panel('nodes')
        self.changes = []
        self.panel.add_checkbox('axes', on_change=self.changes.append)

    def value(self):
        return self.panel._snapshot()['controls'][0]['value']

    def event(self, value, event_id='toggle'):
        return dict(type='ui_event', panel_id='nodes',
                    session=self.panel._snapshot()['session'],
                    event_id=event_id, id='axes', value=value)

    def test_programmatic_updates_preserve_booleans_without_callbacks(self):
        self.assertIs(self.value(), False)
        self.panel.set_value('axes', True)
        self.assertIs(self.value(), True)
        self.panel.set_value('axes', False)
        self.assertIs(self.value(), False)
        self.assertEqual(self.changes, [])
        self.ui.add_checkbox('default_axes', value=True)
        self.assertIs(self.ui._snapshot()['controls'][0]['value'], True)

    def test_non_booleans_are_rejected_at_definition_update_and_dispatch(self):
        for index, value in enumerate((0, 1, 'false', 'true', None, [], {})):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    UIPanel().add_checkbox('axes', value=value)
                with self.assertRaises(ValueError):
                    self.panel.set_value('axes', value)
                with self.assertLogs('wrs.viewer.web_ui.panel', level='WARNING'):
                    result = self.ui._handle_event(self.event(value, f'invalid_{index}'))
                self.assertFalse(result['ok'])
                self.assertIs(self.value(), False)
        self.assertEqual(self.changes, [])

    def test_events_route_boolean_values_and_are_deduplicated(self):
        event = self.event(True)
        result = self.ui._handle_event(event)
        self.assertTrue(result['ok'])
        self.assertIs(self.value(), True)
        self.assertEqual(result['panel_id'], 'nodes')
        self.assertEqual(self.ui._handle_event(event), result)
        self.assertEqual(self.changes, [True])
        result = self.ui._handle_event(self.event(False, 'uncheck'))
        self.assertTrue(result['ok'])
        self.assertIs(self.value(), False)
        self.assertEqual(self.changes, [True, False])

    def test_disabled_checkbox_rejects_events(self):
        self.panel.set_enabled('axes', False)
        with self.assertLogs('wrs.viewer.web_ui.panel', level='WARNING'):
            result = self.ui._handle_event(self.event(True))
        self.assertFalse(result['ok'])
        self.assertIs(self.value(), False)
        self.assertEqual(self.changes, [])

    def test_callback_failure_restores_checked_and_unchecked_values(self):
        def fail(checked):
            raise RuntimeError('Could not change axes')

        self.panel.remove('axes')
        self.panel.add_checkbox('axes', on_change=fail)
        for initial in (False, True):
            with self.subTest(initial=initial):
                self.panel.set_value('axes', initial)
                with self.assertLogs('wrs.viewer.web_ui.panel', level='WARNING'):
                    result = self.ui._handle_event(self.event(not initial, str(initial)))
                self.assertFalse(result['ok'])
                self.assertEqual(result['error'], 'Could not change axes')
                self.assertIs(self.value(), initial)
                self.assertIs(result['state']['controls'][0]['value'], initial)


if __name__ == '__main__':
    unittest.main()
