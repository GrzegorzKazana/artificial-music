import mido


def create_int_coinstrainer(lo, hi): return lambda n: min(max(lo, n), hi)


constraint_note = create_int_coinstrainer(0, 127)


def transpose_midi(step, track):
    def handle_msg(msg):
        return (msg.copy(note=constraint_note(msg.note + step))
                if msg.type in ['note_on', 'note_off'] else msg)

    return [handle_msg(m) for m in track]
