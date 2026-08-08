import tempfile
import unittest
from pathlib import Path

from src.creator_learning.workflow_checkpoint import (
    find_checkpoint,
    list_all_checkpoints,
    list_checkpoints,
    load_checkpoint,
    load_latest_checkpoint,
    save_checkpoint,
    validate_checkpoint,
    validate_checkpoints,
)
from src.creator_learning.workflow_exceptions import WorkflowCheckpointError, WorkflowCheckpointMismatchError
from src.creator_learning.workflow_models import WorkflowCheckpoint, new_workflow_id
from src.creator_learning.workflow_state import WorkflowState, validate_transition
from src.creator_learning.workflow_exceptions import InvalidWorkflowTransitionError


def _checkpoint(**overrides):
    defaults = dict(
        workflow_id=new_workflow_id(), creator_id="c1", platform="instagram", username="demo",
        profile_url="https://x/demo", connector_name="dummy",
    )
    defaults.update(overrides)
    return WorkflowCheckpoint(**defaults)


class WorkflowStateTransitionTests(unittest.TestCase):
    def test_forward_progression_allowed(self):
        order = WorkflowState.ORDER
        for current, target in zip(order, order[1:]):
            validate_transition(current, target)  # must not raise

    def test_failed_reachable_from_any_non_terminal_state(self):
        for state in WorkflowState.ORDER[:-1]:
            validate_transition(state, WorkflowState.FAILED)

    def test_cancelled_reachable_from_any_non_terminal_state(self):
        for state in WorkflowState.ORDER[:-1]:
            validate_transition(state, WorkflowState.CANCELLED)

    def test_skipping_a_state_rejected(self):
        with self.assertRaises(InvalidWorkflowTransitionError):
            validate_transition(WorkflowState.CREATED, WorkflowState.LEARNING)

    def test_backward_move_rejected(self):
        with self.assertRaises(InvalidWorkflowTransitionError):
            validate_transition(WorkflowState.LEARNING, WorkflowState.RESEARCHING)

    def test_terminal_states_have_no_outgoing_transitions(self):
        for terminal in WorkflowState.TERMINAL:
            for target in WorkflowState.ALL:
                with self.assertRaises(InvalidWorkflowTransitionError):
                    validate_transition(terminal, target)

    def test_unrecognized_state_rejected(self):
        with self.assertRaises(InvalidWorkflowTransitionError):
            validate_transition("not_a_real_state", WorkflowState.RESEARCHING)


class WorkflowCheckpointModelTests(unittest.TestCase):
    def test_record_transition_updates_state_and_last_completed_step(self):
        checkpoint = _checkpoint()
        checkpoint.record_transition(WorkflowState.RESEARCHING)
        self.assertEqual(checkpoint.state, WorkflowState.RESEARCHING)
        self.assertEqual(checkpoint.last_completed_step, WorkflowState.RESEARCHING)
        self.assertEqual(len(checkpoint.step_history), 1)

    def test_record_transition_to_failed_does_not_change_last_completed_step(self):
        checkpoint = _checkpoint()
        checkpoint.record_transition(WorkflowState.RESEARCHING)
        checkpoint.record_transition(WorkflowState.FAILED)
        self.assertEqual(checkpoint.state, WorkflowState.FAILED)
        self.assertEqual(checkpoint.last_completed_step, WorkflowState.RESEARCHING)

    def test_record_transition_mints_a_fresh_resume_token(self):
        checkpoint = _checkpoint()
        first_token = checkpoint.resume_token
        checkpoint.record_transition(WorkflowState.RESEARCHING)
        self.assertNotEqual(checkpoint.resume_token, first_token)

    def test_to_dict_from_dict_round_trip(self):
        checkpoint = _checkpoint()
        checkpoint.record_transition(WorkflowState.RESEARCHING)
        restored = WorkflowCheckpoint.from_dict(checkpoint.to_dict())
        self.assertEqual(checkpoint, restored)


class SaveLoadCheckpointTests(unittest.TestCase):
    def test_save_and_load_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = _checkpoint()
            save_checkpoint(Path(tmp), checkpoint)
            loaded = load_checkpoint(Path(tmp), "c1", checkpoint.workflow_id)
            self.assertEqual(loaded, checkpoint)

    def test_missing_checkpoint_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_checkpoint(Path(tmp), "c1", "nonexistent"))

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c1" / "workflow" / "abc.json"
            path.parent.mkdir(parents=True)
            path.write_text("{not valid")
            with self.assertRaises(WorkflowCheckpointError):
                load_checkpoint(Path(tmp), "c1", "abc")

    def test_save_updates_latest_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = _checkpoint()
            save_checkpoint(Path(tmp), first)
            second = _checkpoint()
            save_checkpoint(Path(tmp), second)
            latest = load_latest_checkpoint(Path(tmp), "c1")
            self.assertEqual(latest.workflow_id, second.workflow_id)

    def test_latest_pointer_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_latest_checkpoint(Path(tmp), "no_such_creator"))


class ListAndFindCheckpointTests(unittest.TestCase):
    def test_list_checkpoints_excludes_latest_pointer_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp), _checkpoint())
            save_checkpoint(Path(tmp), _checkpoint())
            checkpoints = list_checkpoints(Path(tmp), "c1")
            self.assertEqual(len(checkpoints), 2)

    def test_list_checkpoints_empty_for_unknown_creator(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_checkpoints(Path(tmp), "nobody"), [])

    def test_list_all_checkpoints_spans_creators(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp), _checkpoint(creator_id="c1"))
            save_checkpoint(Path(tmp), _checkpoint(creator_id="c2"))
            self.assertEqual(len(list_all_checkpoints(Path(tmp))), 2)

    def test_find_checkpoint_by_id_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = _checkpoint(creator_id="c_unusual")
            save_checkpoint(Path(tmp), checkpoint)
            found = find_checkpoint(Path(tmp), checkpoint.workflow_id)
            self.assertEqual(found.creator_id, "c_unusual")

    def test_find_checkpoint_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_checkpoint(Path(tmp), "nonexistent"))


class ValidateCheckpointTests(unittest.TestCase):
    def test_matching_identity_does_not_raise(self):
        checkpoint = _checkpoint()
        validate_checkpoint(checkpoint, workflow_id=checkpoint.workflow_id, creator_id="c1")

    def test_workflow_id_mismatch_raises(self):
        checkpoint = _checkpoint()
        with self.assertRaises(WorkflowCheckpointMismatchError):
            validate_checkpoint(checkpoint, workflow_id="other", creator_id="c1")

    def test_creator_id_mismatch_raises(self):
        checkpoint = _checkpoint()
        with self.assertRaises(WorkflowCheckpointMismatchError):
            validate_checkpoint(checkpoint, workflow_id=checkpoint.workflow_id, creator_id="other")


class ValidateCheckpointsStructuralTests(unittest.TestCase):
    def test_passes_on_clean_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp), _checkpoint())
            passed, errors = validate_checkpoints(Path(tmp))
            self.assertTrue(passed)
            self.assertEqual(errors, [])

    def test_empty_directory_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            passed, errors = validate_checkpoints(Path(tmp))
            self.assertTrue(passed)

    def test_unrecognized_state_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = _checkpoint()
            checkpoint.state = "not_a_real_state"
            save_checkpoint(Path(tmp), checkpoint)
            passed, errors = validate_checkpoints(Path(tmp))
            self.assertFalse(passed)
            self.assertTrue(any("unrecognized state" in error for error in errors))

    def test_scoping_to_one_creator(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp), _checkpoint(creator_id="c1"))
            save_checkpoint(Path(tmp), _checkpoint(creator_id="c2"))
            passed, errors = validate_checkpoints(Path(tmp), "c1")
            self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
